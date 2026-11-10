#![cfg(all(feature = "cpu", feature = "ready"))]

//! E/P/D failure, cancellation and cleanup ownership: timing tests.
//!
//! An E/P/D request moves through stages. Encoders produce embeddings, an EC
//! transfer carries them to the consumer (Prefill, or the combined worker),
//! and Decode produces the reply. `EcTransferGuard`
//! (`src/routers/http/epd.rs`) is armed when the transfers start. While it is
//! armed, a failure or a dropped request makes the router cancel the
//! transfers. A handoff event disarms it; from then on the consumer owns the
//! cleanup. These tests pin down what happens when a failure, a stall or a
//! disconnect lands on either side of that event.
//!
//! Four timing cases:
//!
//! 1. an early encoder failure while a sibling encoder is still blocked;
//! 2. a consumer that sends 2xx headers, then a stalled or invalid body;
//! 3. a Decode disconnect, before the first byte or in the middle of SSE;
//! 4. a client disconnect that races a handoff event.
//!
//! Each case runs on three paths. They disarm the guard at different handoff
//! events:
//!
//! - `regular_e_pd`: at the combined worker's 2xx response headers
//!   (`Router::route_transparent`);
//! - `sequential_e_p_d`: after Prefill's whole body, its JSON parse and a KV
//!   check (`VllmPDRouter::process_vllm_two_stage_request`);
//! - `concurrent_nixl`: after a join of Prefill's whole body with Decode's
//!   headers, and an optional parse (`process_concurrent_two_stage`).
//!
//! "The request ended" can mean five different things, and they do not always
//! happen together: HTTP socket disconnect, local future or stream drop,
//! encoder HTTP abort, EC control cancellation and GPU completion. Each
//! assertion names the one it checks:
//!
//! - `[local drop]`: the test drops the router's request future or its
//!   response body. Hyper does the same when a client socket closes; the
//!   client socket itself is not part of these tests.
//! - `[socket close]`: a Prefill, Decode or combined worker mock sees its open
//!   request end early, because the router closed the connection.
//! - `[encoder abort]`: the same, seen by an encoder mock.
//! - `[EC cancel]`: the consumer's EC control address, a ZMQ REP recorder,
//!   gets `peers` and then `cancel` with `abandon: true`.
//! - GPU completion needs real engines; no test here sees it.
//!
//! Two more labels support them: `[client]` for what the caller of the router
//! gets, and `[dispatch]` for which mocks got a request.
//!
//! The harness lives in `common/epd/`. Gates set the timing, not sleeps. Each
//! mock signals a `Notify` when a request arrives, and holds its reply on a
//! `Semaphore` until the test adds a permit. Only checks that something does
//! *not* happen wait for a fixed time (`QUIET`): such a check can miss a late
//! event, but it cannot fail while the behavior is as described.
//!
//! Run: `cargo test --features cpu,ready --test test_epd_failure_timing`.

#[path = "common/epd/rig.rs"]
mod epd;

use axum::http::StatusCode;
use epd::*;
use serde_json::Value;
use std::time::{Duration, Instant};

/// An early encoder failure while a sibling encoder is still blocked.
///
/// `EncoderStage::prepare` sends one request per encoder batch and waits for
/// all of them with `join_all`. Only then does it return the first error, in
/// batch order. The encoder stage runs before the sequential or concurrent
/// branch, so the three paths behave the same.
mod early_encoder_failure_with_blocked_sibling {
    use super::*;
    use axum::response::Response;
    use tokio::task::JoinHandle;

    /// Sends a request where E0 fails at once with 503 and E1 holds its
    /// reply. Returns when E0 has answered and E1 has the request.
    async fn e0_failed_e1_blocked(path: Path) -> (Rig, JoinHandle<Response>) {
        let rig = Rig::new(path).await;
        rig.e0.set_mode(Mode::FailWithStatus(503));
        rig.e1.set_mode(Mode::HoldThenReply);
        let client = rig.send(two_images(false));
        rig.e0.expect_completed().await;
        rig.e1.wait_requests(1).await;
        rig.expect_encoder_assignment();
        (rig, client)
    }

    /// E0 fails at once; E1 is blocked and answers in the end.
    async fn sibling_answers_late(path: Path) {
        let (mut rig, mut client) = e0_failed_e1_blocked(path).await;
        // gap: the failure is known, but join_all keeps the request open and
        // E1 working. A sibling abort (try_join_all) would end both here.
        expect_waiting(&mut client).await;
        rig.e1.expect_open();
        // gap: the armed guard lives in prepare, so the cancel waits for E1.
        rig.ec.expect_no_cancel().await;
        rig.e1.release();
        let response = answer(client).await;
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "[client] E0's error"
        );
        rig.e1.expect_completed().await;
        let ids = rig.transfer_ids();
        rig.ec.expect_cancel(&ids).await;
        rig.consumer.expect_no_request();
        rig.decode.expect_no_request();
    }

    /// E0 fails at once; the client goes away while E1 is still blocked.
    async fn client_leaves_while_sibling_blocked(path: Path) {
        let (mut rig, mut client) = e0_failed_e1_blocked(path).await;
        expect_waiting(&mut client).await;
        disconnect(client).await;
        // The drop reaches join_all: E1's connection closes, and the armed
        // guard cancels both transfers.
        rig.e1.expect_aborted().await;
        let ids = rig.transfer_ids();
        rig.ec.expect_cancel(&ids).await;
        rig.consumer.expect_no_request();
        rig.decode.expect_no_request();
    }

    #[tokio::test]
    async fn regular_e_pd() {
        sibling_answers_late(Path::CombinedPd).await;
        client_leaves_while_sibling_blocked(Path::CombinedPd).await;
    }

    #[tokio::test]
    async fn sequential_e_p_d() {
        sibling_answers_late(Path::SplitPdSequential).await;
        client_leaves_while_sibling_blocked(Path::SplitPdSequential).await;
    }

    #[tokio::test]
    async fn concurrent_nixl() {
        sibling_answers_late(Path::SplitPdConcurrent).await;
        client_leaves_while_sibling_blocked(Path::SplitPdConcurrent).await;
    }

    /// Only the router's HTTP client timeout bounds that wait.
    #[tokio::test]
    async fn blocked_sibling_holds_the_request_until_the_client_timeout() {
        let timeout = Duration::from_secs(1);
        let options = Options {
            client_timeout: Some(timeout),
            ..Options::default()
        };
        let mut rig = Rig::with(Path::CombinedPd, options).await;
        rig.e0.set_mode(Mode::FailWithStatus(503));
        rig.e1.set_mode(Mode::HoldThenReply); // never released
        let started = Instant::now();
        let response = answer(rig.send(two_images(false))).await;
        // gap: with the default timeout, this answer takes 30 minutes.
        assert!(
            started.elapsed() >= timeout,
            "[client] answered after {:?}, before the client timeout",
            started.elapsed()
        );
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "[client] E0's error, not E1's timeout"
        );
        // The timeout, not the failure, ended E1's request.
        rig.e1.expect_aborted().await;
        let ids = rig.transfer_ids();
        rig.ec.expect_cancel(&ids).await;
        rig.consumer.expect_no_request();
    }

    /// `prepare` returns the first error in batch order, not the first in
    /// time. A sibling abort would return E1's error instead, so this test
    /// changes with that patch.
    #[tokio::test]
    async fn first_error_in_batch_order_wins() {
        let mut rig = Rig::new(Path::CombinedPd).await;
        rig.e0.set_mode(Mode::HoldThenFailWithStatus(500));
        rig.e1.set_mode(Mode::FailWithStatus(503));
        let mut client = rig.send(two_images(false));
        rig.e1.expect_completed().await;
        rig.e0.wait_requests(1).await;
        rig.expect_encoder_assignment();
        expect_waiting(&mut client).await;
        rig.e0.release();
        let response = answer(client).await;
        assert_eq!(
            response.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "[client] E0's 500 (batch 0), not E1's earlier 503"
        );
        let ids = rig.transfer_ids();
        rig.ec.expect_cancel(&ids).await;
        rig.consumer.expect_no_request();
    }
}

/// The EC consumer sends 2xx headers, then its body stalls or is invalid.
mod consumer_2xx_with_stalled_or_invalid_body {
    use super::*;

    /// E+PD hands off at PD's 2xx headers, so a stalled or invalid body comes
    /// after the handoff.
    #[tokio::test]
    async fn regular_e_pd() {
        // The body completes, but it is not JSON.
        {
            let mut rig = Rig::new(Path::CombinedPd).await;
            rig.consumer.set_mode(Mode::StallBodyInvalidJson);
            let response = answer(rig.send(two_images(false))).await;
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "[client] the 2xx headers come through at once"
            );
            let mut body = response.into_body().into_data_stream();
            assert_eq!(next_chunk(&mut body).await, "{", "[client] first part");
            // Disarmed at the headers: nothing to cancel while the body stalls.
            rig.ec.expect_no_cancel().await;
            rig.consumer.release();
            let (rest, broken) = rest_of_body(body).await;
            assert!(!broken, "[client] the body ends normally");
            assert!(
                serde_json::from_str::<Value>(&format!("{{{rest}")).is_err(),
                "[client] the invalid body comes through as it is: {rest:?}"
            );
            rig.ec.expect_no_cancel().await;
        }
        // The client goes away while the body stalls.
        {
            let mut rig = Rig::new(Path::CombinedPd).await;
            rig.consumer.set_mode(Mode::StallBody);
            let response = answer(rig.send(two_images(false))).await;
            let mut body = response.into_body().into_data_stream();
            next_chunk(&mut body).await;
            drop(body); // [local drop]: the response body
            rig.consumer.expect_aborted().await;
            rig.ec.expect_no_cancel().await;
        }
    }

    /// Sequential E/P/D reads P's whole body, parses it and checks for KV
    /// parameters before it disarms. Each failure drops an armed guard, and D
    /// is never called.
    #[tokio::test]
    async fn sequential_e_p_d() {
        for (mode, error) in [
            (
                Mode::StallBodyInvalidJson,
                "Failed to parse prefill response as JSON",
            ),
            (Mode::StallBodyWithoutKv, "refusing metadata-only decode"),
            (Mode::BreakBody, "Failed to read response body"),
        ] {
            let mut rig = Rig::new(Path::SplitPdSequential).await;
            rig.consumer.set_mode(mode);
            let mut client = rig.send(two_images(false));
            rig.consumer.wait_requests(1).await;
            // While P's body stalls, the guard is armed and D waits.
            expect_waiting(&mut client).await;
            rig.ec.expect_no_cancel().await;
            rig.decode.expect_no_request();
            rig.consumer.release();
            let response = answer(client).await;
            assert_eq!(
                response.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "[client] {mode:?}"
            );
            let text = text(response).await;
            assert!(text.contains(error), "[client] {text}");
            let ids = rig.transfer_ids();
            rig.ec.expect_cancel(&ids).await;
            rig.decode.expect_no_request();
        }
        // The client goes away while P's body stalls.
        let mut rig = Rig::new(Path::SplitPdSequential).await;
        rig.consumer.set_mode(Mode::StallBody);
        let mut client = rig.send(two_images(false));
        rig.consumer.wait_requests(1).await;
        expect_waiting(&mut client).await;
        disconnect(client).await;
        rig.consumer.expect_aborted().await;
        let ids = rig.transfer_ids();
        rig.ec.expect_cancel(&ids).await;
        rig.decode.expect_no_request();
    }

    /// The concurrent path joins P's whole body with D's headers, parses P's
    /// body with `.ok()` and disarms even when that gives nothing. It has no
    /// KV check, and D is already running.
    #[tokio::test]
    async fn concurrent_nixl() {
        for mode in [Mode::StallBodyInvalidJson, Mode::StallBodyWithoutKv] {
            let mut rig = Rig::new(Path::SplitPdConcurrent).await;
            rig.consumer.set_mode(mode);
            let mut client = rig.send(two_images(false));
            rig.consumer.wait_requests(1).await;
            // [dispatch] D got its request at the same time as P, and answered.
            rig.decode.expect_completed().await;
            expect_waiting(&mut client).await;
            rig.ec.expect_no_cancel().await;
            rig.consumer.release();
            let response = answer(client).await;
            // gap: P's body is not usable, but the router disarms and returns
            // D's reply. The sequential path answers 500 and cancels instead.
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "[client] D's reply, with P in {mode:?}"
            );
            let reply: Value = serde_json::from_str(&text(response).await).unwrap();
            assert_eq!(
                reply["choices"][0]["message"]["content"], "A",
                "[client] D's reply"
            );
            rig.ec.expect_no_cancel().await;
        }
        // The client goes away while P's body stalls. D has sent its headers
        // and holds its body.
        let mut rig = Rig::new(Path::SplitPdConcurrent).await;
        rig.consumer.set_mode(Mode::StallBody);
        rig.decode.set_mode(Mode::StallBody);
        let mut client = rig.send(two_images(false));
        rig.consumer.wait_requests(1).await;
        rig.decode.wait_requests(1).await;
        expect_waiting(&mut client).await;
        disconnect(client).await;
        rig.consumer.expect_aborted().await;
        // D's cleanup is a separate observation: its connection closes too.
        rig.decode.expect_aborted().await;
        let ids = rig.transfer_ids();
        rig.ec.expect_cancel(&ids).await;
    }
}

/// D disconnects, before its first byte or in the middle of an SSE
/// stream. On E+PD, PD has both roles.
mod decode_disconnect {
    use super::*;

    fn raw_last_stage() -> Options {
        Options {
            raw_last_stage: true,
            ..Options::default()
        }
    }

    /// D sends its headers and one SSE event, then its stream breaks. This is
    /// after the handoff on every path, so the router sends no cancel.
    /// Whether the consumer frees its EC buffer (GPU completion) is not
    /// visible here.
    async fn decode_breaks_mid_sse(path: Path) {
        let mut rig = Rig::new(path).await;
        let decode = if path == Path::CombinedPd {
            &rig.consumer
        } else {
            &rig.decode
        };
        decode.set_mode(Mode::BreakBody);
        let response = answer(rig.send(two_images(true))).await;
        assert_eq!(response.status(), StatusCode::OK, "[client]");
        let mut body = response.into_body().into_data_stream();
        let first = next_chunk(&mut body).await;
        assert!(first.starts_with("data: "), "[client] first event: {first}");
        decode.release();
        let (_, broken) = rest_of_body(body).await;
        assert!(broken, "[client] the stream ends with an error");
        rig.ec.expect_no_cancel().await;
    }

    #[tokio::test]
    async fn regular_e_pd() {
        // Before the first byte: a transport error comes before the handoff,
        // so the armed guard cancels.
        {
            let mut rig = Rig::with(Path::CombinedPd, raw_last_stage()).await;
            rig.consumer.set_mode(Mode::HangUp);
            let client = rig.send(two_images(true));
            rig.consumer.wait_requests(1).await;
            rig.consumer.release();
            let response = answer(client).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_GATEWAY,
                "[client] transport error"
            );
            let ids = rig.transfer_ids();
            rig.ec.expect_cancel(&ids).await;
        }
        decode_breaks_mid_sse(Path::CombinedPd).await;
    }

    #[tokio::test]
    async fn sequential_e_p_d() {
        // Before the first byte.
        {
            let mut rig = Rig::with(Path::SplitPdSequential, raw_last_stage()).await;
            rig.decode.set_mode(Mode::HangUp);
            let client = rig.send(two_images(true));
            rig.decode.wait_requests(1).await;
            rig.consumer.expect_completed().await;
            rig.decode.release();
            let response = answer(client).await;
            assert_eq!(
                response.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "[client]"
            );
            let text = text(response).await;
            assert!(text.contains("Decode request failed"), "[client] {text}");
            // P has used the embeddings, and the guard was disarmed before D
            // was sent: no EC cancel. What happens to P's KV is a separate
            // question.
            rig.ec.expect_no_cancel().await;
        }
        decode_breaks_mid_sse(Path::SplitPdSequential).await;
    }

    #[tokio::test]
    async fn concurrent_nixl() {
        // Before the first byte, while P is still working.
        {
            let mut rig = Rig::with(Path::SplitPdConcurrent, raw_last_stage()).await;
            rig.consumer.set_mode(Mode::HoldThenReply);
            rig.decode.set_mode(Mode::HangUp);
            let mut client = rig.send(two_images(true));
            rig.consumer.wait_requests(1).await;
            rig.decode.wait_requests(1).await;
            rig.decode.release();
            rig.decode.expect_completed().await;
            // gap: tokio::join! waits for P's whole reply, so D's failure
            // neither answers the client nor stops P.
            expect_waiting(&mut client).await;
            rig.consumer.expect_open();
            rig.ec.expect_no_cancel().await;
            rig.consumer.release();
            let response = answer(client).await;
            assert_eq!(
                response.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "[client]"
            );
            let text = text(response).await;
            assert!(text.contains("Decode request failed"), "[client] {text}");
            // P answered 2xx, so the guard was disarmed after the join.
            rig.ec.expect_no_cancel().await;
        }
        decode_breaks_mid_sse(Path::SplitPdConcurrent).await;
    }
}

/// The client goes away just before or just after a handoff event.
/// Gates put the drop on one side of the event, so the result does not depend
/// on which one runs first.
mod client_drop_races_the_handoff {
    use super::*;

    #[tokio::test]
    async fn regular_e_pd() {
        // Just before: PD has the request, but has not sent its headers.
        {
            let mut rig = Rig::new(Path::CombinedPd).await;
            rig.consumer.set_mode(Mode::HoldThenReply);
            let mut client = rig.send(two_images(false));
            rig.consumer.wait_requests(1).await;
            expect_waiting(&mut client).await;
            disconnect(client).await;
            rig.consumer.expect_aborted().await;
            // gap: PD already holds the request, but the router cancels its
            // transfers. A non-streaming reply sends its headers only when it
            // is complete, so this window lasts for the whole generation.
            let ids = rig.transfer_ids();
            rig.ec.expect_cancel(&ids).await;
        }
        // Just after: PD sent its 2xx headers and streams.
        {
            let mut rig = Rig::new(Path::CombinedPd).await;
            rig.consumer.set_mode(Mode::StallBody);
            let response = answer(rig.send(two_images(true))).await;
            assert_eq!(response.status(), StatusCode::OK, "[client]");
            let mut body = response.into_body().into_data_stream();
            next_chunk(&mut body).await;
            drop(body); // [local drop]: the response body
            rig.consumer.expect_aborted().await;
            rig.ec.expect_no_cancel().await;
        }
    }

    #[tokio::test]
    async fn sequential_e_p_d() {
        // Just before: P has the request.
        {
            let mut rig = Rig::new(Path::SplitPdSequential).await;
            rig.consumer.set_mode(Mode::HoldThenReply);
            let mut client = rig.send(two_images(false));
            rig.consumer.wait_requests(1).await;
            expect_waiting(&mut client).await;
            disconnect(client).await;
            rig.consumer.expect_aborted().await;
            let ids = rig.transfer_ids();
            rig.ec.expect_cancel(&ids).await;
            rig.decode.expect_no_request();
        }
        // Just after: P is done, and D has the request.
        {
            let mut rig = Rig::new(Path::SplitPdSequential).await;
            rig.decode.set_mode(Mode::HoldThenReply);
            let mut client = rig.send(two_images(false));
            rig.decode.wait_requests(1).await;
            rig.consumer.expect_completed().await;
            expect_waiting(&mut client).await;
            disconnect(client).await;
            rig.decode.expect_aborted().await;
            rig.ec.expect_no_cancel().await;
        }
    }

    #[tokio::test]
    async fn concurrent_nixl() {
        // Just before: P and D both have the request.
        {
            let mut rig = Rig::new(Path::SplitPdConcurrent).await;
            rig.consumer.set_mode(Mode::HoldThenReply);
            rig.decode.set_mode(Mode::HoldThenReply);
            let mut client = rig.send(two_images(false));
            rig.consumer.wait_requests(1).await;
            // [dispatch] D gets its request while P holds: the concurrent path.
            rig.decode.wait_requests(1).await;
            expect_waiting(&mut client).await;
            disconnect(client).await;
            rig.consumer.expect_aborted().await;
            rig.decode.expect_aborted().await;
            let ids = rig.transfer_ids();
            rig.ec.expect_cancel(&ids).await;
        }
        // P has sent its whole 2xx reply; D has not sent its headers.
        {
            let mut rig = Rig::new(Path::SplitPdConcurrent).await;
            rig.decode.set_mode(Mode::HoldThenReply);
            let mut client = rig.send(two_images(false));
            rig.consumer.expect_completed().await;
            rig.decode.wait_requests(1).await;
            expect_waiting(&mut client).await;
            disconnect(client).await;
            rig.decode.expect_aborted().await;
            // gap: this path disarms only when tokio::join! has both P's body
            // and D's headers. P is done, but the router cancels its
            // transfers; at the same point the sequential path has disarmed.
            // A non-streaming D sends its headers only with the whole reply.
            let ids = rig.transfer_ids();
            rig.ec.expect_cancel(&ids).await;
        }
        // Just after: D sent its headers and streams.
        {
            let mut rig = Rig::new(Path::SplitPdConcurrent).await;
            rig.decode.set_mode(Mode::StallBody);
            let response = answer(rig.send(two_images(true))).await;
            assert_eq!(response.status(), StatusCode::OK, "[client]");
            let mut body = response.into_body().into_data_stream();
            next_chunk(&mut body).await;
            drop(body); // [local drop]: the response body
            rig.decode.expect_aborted().await;
            rig.ec.expect_no_cancel().await;
        }
    }
}
