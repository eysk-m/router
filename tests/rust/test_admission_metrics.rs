#![cfg(all(feature = "cpu", feature = "ready"))]

//! What the concurrency limiter reports when it cannot admit a request at once:
//! the reason header on each rejection and the `vllm_router_admission_*` series.
//!
//! A limiter with no tokens sends every request down the slow path. Each test
//! runs on one thread with its own recorder, so it sees only its own series.

mod common;

use axum::body::Body;
use axum::http::header::RETRY_AFTER;
use axum::http::{Response, StatusCode};
use axum::Router;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tower::ServiceExt;
use vllm_router_rs::config::RouterConfig;
use vllm_router_rs::middleware::{ConcurrencyLimiter, QueuedRequest, ADMISSION_REASON_HEADER};
use vllm_router_rs::routers::{RouterFactory, RouterTrait};
use vllm_router_rs::server::{build_app, AppContext, AppState};

const NAMED: &str = "/v1/completions";
const UNMATCHED: &str = "/no/such/route";

/// Run `test` on one thread with its own recorder.
fn with_metrics<F, Fut>(test: F)
where
    F: FnOnce(PrometheusHandle) -> Fut,
    Fut: Future<Output = ()>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    metrics::with_local_recorder(&recorder, || runtime.block_on(test(handle)));
}

fn context(tokens: usize) -> Arc<AppContext> {
    common::create_test_context(RouterConfig {
        max_concurrent_requests: tokens,
        ..Default::default()
    })
}

async fn app(context: &Arc<AppContext>, queue_tx: Option<mpsc::Sender<QueuedRequest>>) -> Router {
    let router: Arc<dyn RouterTrait> =
        Arc::from(RouterFactory::create_router(context).await.expect("router"));
    let state = Arc::new(AppState {
        router,
        context: context.clone(),
        concurrency_queue_tx: queue_tx,
        router_manager: None,
    });
    build_app(
        state,
        context.router_config.max_payload_size,
        vec![],
        vec![],
        true,
    )
}

/// A queue with a running processor.
fn running_queue(context: &AppContext, timeout: Duration) -> mpsc::Sender<QueuedRequest> {
    let (limiter, processor) = ConcurrencyLimiter::new(context.rate_limiter.clone(), 1, timeout);
    tokio::spawn(processor.expect("processor").run());
    limiter.queue_tx.expect("queue")
}

async fn send(app: &Router, path: &str) -> Response<Body> {
    app.clone()
        .oneshot(common::http::post(path, "{}".to_string()))
        .await
        .expect("response")
}

/// Send a request that waits in the queue, and return once the gauge counts it.
async fn park(app: &Router, metrics: &PrometheusHandle) -> JoinHandle<Response<Body>> {
    let app = app.clone();
    let request = tokio::spawn(async move { send(&app, UNMATCHED).await });
    wait_for_line(metrics, "vllm_router_admission_waiting 1").await;
    request
}

fn has_line(metrics: &PrometheusHandle, line: &str) -> bool {
    metrics.render().lines().any(|rendered| rendered == line)
}

async fn wait_for_line(metrics: &PrometheusHandle, line: &str) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !has_line(metrics, line) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("missing `{line}` in:\n{}", metrics.render()));
}

#[track_caller]
fn assert_line(metrics: &PrometheusHandle, line: &str) {
    assert!(
        has_line(metrics, line),
        "missing `{line}` in:\n{}",
        metrics.render()
    );
}

#[track_caller]
fn assert_no_rejections(metrics: &PrometheusHandle) {
    let rendered = metrics.render();
    assert!(
        !rendered.contains("vllm_router_admission_rejected_total"),
        "{rendered}"
    );
}

#[track_caller]
fn assert_rejected(response: &Response<Body>, status: StatusCode, reason: &str) {
    assert_eq!(response.status(), status);
    assert_eq!(
        response
            .headers()
            .get(ADMISSION_REASON_HEADER)
            .expect("reason header"),
        reason
    );
}

#[track_caller]
fn assert_retry_after(response: &Response<Body>) {
    assert!(response.headers().contains_key(RETRY_AFTER));
}

#[test]
fn immediate_admission_records_nothing() {
    with_metrics(|metrics| async move {
        let app = app(&context(1), None).await;

        let response = send(&app, UNMATCHED).await;

        assert!(!response.headers().contains_key(ADMISSION_REASON_HEADER));
        let rendered = metrics.render();
        assert!(!rendered.contains("vllm_router_admission_"), "{rendered}");
    });
}

#[test]
fn no_capacity_when_queuing_is_disabled() {
    with_metrics(|metrics| async move {
        let app = app(&context(0), None).await;

        for path in [NAMED, UNMATCHED] {
            let response = send(&app, path).await;
            assert_rejected(&response, StatusCode::TOO_MANY_REQUESTS, "no_capacity");
            assert_retry_after(&response);
        }

        assert_line(
            &metrics,
            r#"vllm_router_admission_rejected_total{reason="no_capacity"} 2"#,
        );
    });
}

#[test]
fn queue_full_when_the_queue_has_no_room() {
    with_metrics(|metrics| async move {
        // Nothing reads this queue, so the first request stays in it.
        let (queue_tx, _queue_rx) = mpsc::channel(1);
        let app = app(&context(0), Some(queue_tx)).await;
        let parked = park(&app, &metrics).await;

        for path in [NAMED, UNMATCHED] {
            let response = send(&app, path).await;
            assert_rejected(&response, StatusCode::TOO_MANY_REQUESTS, "queue_full");
            assert_retry_after(&response);
        }

        assert_line(
            &metrics,
            r#"vllm_router_admission_rejected_total{reason="queue_full"} 2"#,
        );
        parked.abort();
    });
}

#[test]
fn queue_timeout_when_no_token_comes_in_time() {
    with_metrics(|metrics| async move {
        let context = context(0);
        let queue_tx = running_queue(&context, Duration::from_millis(10));
        let app = app(&context, Some(queue_tx)).await;

        for path in [NAMED, UNMATCHED] {
            let response = send(&app, path).await;
            assert_rejected(&response, StatusCode::REQUEST_TIMEOUT, "queue_timeout");
        }

        assert_line(
            &metrics,
            r#"vllm_router_admission_rejected_total{reason="queue_timeout"} 2"#,
        );
        assert_line(&metrics, "vllm_router_admission_waiting 0");
        assert_line(&metrics, "vllm_router_admission_wait_seconds_count 2");
    });
}

#[test]
fn closed_when_the_queue_stops() {
    with_metrics(|metrics| async move {
        let (queue_tx, queue_rx) = mpsc::channel(1);
        let app = app(&context(0), Some(queue_tx)).await;
        let parked = park(&app, &metrics).await;

        // A request in the queue loses its reply channel.
        drop(queue_rx);
        let response = parked.await.expect("request");
        assert_rejected(&response, StatusCode::INTERNAL_SERVER_ERROR, "closed");

        // A new request cannot enter the queue.
        let response = send(&app, NAMED).await;
        assert_rejected(&response, StatusCode::TOO_MANY_REQUESTS, "closed");
        assert_retry_after(&response);

        assert_line(
            &metrics,
            r#"vllm_router_admission_rejected_total{reason="closed"} 2"#,
        );
        assert_line(&metrics, "vllm_router_admission_waiting 0");
        assert_line(&metrics, "vllm_router_admission_wait_seconds_count 1");
    });
}

#[test]
fn admitted_request_leaves_the_queue() {
    with_metrics(|metrics| async move {
        let context = context(1);
        let queue_tx = running_queue(&context, Duration::from_secs(60));
        let app = app(&context, Some(queue_tx)).await;

        context.rate_limiter.try_acquire(1.0).await.expect("token");
        let parked = park(&app, &metrics).await;
        context.rate_limiter.return_tokens(1.0).await;

        // The router has no workers, so a 503 means the proxy got the request.
        let response = parked.await.expect("request");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!response.headers().contains_key(ADMISSION_REASON_HEADER));
        assert_line(&metrics, "vllm_router_admission_waiting 0");
        assert_line(&metrics, "vllm_router_admission_wait_seconds_count 1");
        assert_no_rejections(&metrics);
    });
}

/// Hyper drops the handler future when a client disconnects. Aborting the task
/// does the same.
#[test]
fn client_disconnect_leaves_the_queue() {
    with_metrics(|metrics| async move {
        let (queue_tx, _queue_rx) = mpsc::channel(1);
        let app = app(&context(0), Some(queue_tx)).await;

        park(&app, &metrics).await.abort();

        wait_for_line(&metrics, "vllm_router_admission_waiting 0").await;
        assert_line(&metrics, "vllm_router_admission_wait_seconds_count 1");
        assert_no_rejections(&metrics);
    });
}
