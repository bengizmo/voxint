"""The ASGI request-body size cap (`_RequestSizeLimitMiddleware`).

Driven directly at the ASGI layer with controlled ``receive`` frames so the
streaming cap (finding D3 — bound the body regardless of ``Content-Length``) is
exercised deterministically, including the single-response guarantee on overflow.
"""

from starlette.requests import ClientDisconnect

from voxint.api.app import _RequestSizeLimitMiddleware


async def _drive(
    *,
    max_bytes: int,
    headers: list[tuple[bytes, bytes]],
    body_frames: list[dict],
) -> tuple[int | None, bool, int]:
    """Run one HTTP request through the middleware.

    Returns ``(status, inner_ran, response_count)`` where ``inner_ran`` is whether
    the wrapped app produced its own response and ``response_count`` is how many
    ``http.response.start`` messages reached the server (must be exactly 1).
    """
    scope = {"type": "http", "method": "POST", "path": "/x", "headers": headers}
    pending = list(body_frames)
    sent: list[dict] = []
    inner_ran = {"value": False}

    async def receive() -> dict:
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def app(scope: dict, receive_: object, send_: object) -> None:
        # Mimic Starlette's body reader: drain until end-of-body, raising on a
        # disconnect (which is how the truncated over-cap stream surfaces).
        while True:
            message = await receive_()  # type: ignore[operator]
            if message["type"] == "http.disconnect":
                raise ClientDisconnect
            if message["type"] == "http.request" and not message.get("more_body", False):
                break
        inner_ran["value"] = True
        await send_({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[operator]
        await send_({"type": "http.response.body", "body": b"ok"})  # type: ignore[operator]

    async def send(message: dict) -> None:
        sent.append(message)

    middleware = _RequestSizeLimitMiddleware(app, max_bytes=max_bytes)
    await middleware(scope, receive, send)

    starts = [m for m in sent if m["type"] == "http.response.start"]
    status = starts[0]["status"] if starts else None
    return status, inner_ran["value"], len(starts)


def _frame(body: bytes, *, more: bool) -> dict:
    return {"type": "http.request", "body": body, "more_body": more}


async def test_body_at_cap_passes() -> None:
    # Exactly max_bytes, delivered in fragments → the app runs and responds.
    status, inner_ran, count = await _drive(
        max_bytes=100,
        headers=[],  # no Content-Length: the streaming cap is authoritative
        body_frames=[_frame(b"x" * 60, more=True), _frame(b"y" * 40, more=False)],
    )
    assert status == 200
    assert inner_ran is True
    assert count == 1


async def test_one_byte_over_cap_is_rejected_without_content_length() -> None:
    # A chunked/understated body (no Content-Length) one byte over the cap is cut
    # off mid-stream and refused with 413 — the D3 residual, now closed.
    status, inner_ran, count = await _drive(
        max_bytes=100,
        headers=[],
        body_frames=[_frame(b"x" * 60, more=True), _frame(b"y" * 41, more=False)],
    )
    assert status == 413
    assert inner_ran is False  # the body was never fully spooled to the app
    assert count == 1  # exactly one response — no double-send


async def test_single_oversized_frame_is_rejected() -> None:
    # Even when the whole over-cap body arrives in one message, it is refused.
    status, inner_ran, count = await _drive(
        max_bytes=100,
        headers=[],
        body_frames=[_frame(b"z" * 500, more=False)],
    )
    assert status == 413
    assert inner_ran is False
    assert count == 1


async def test_app_error_while_rejecting_still_yields_single_413() -> None:
    # If the app raises something OTHER than ClientDisconnect while unwinding the
    # truncated over-cap stream (e.g. a parser error), the 413 must still be the
    # single response — the exception must not escape once we are rejecting.
    scope = {"type": "http", "method": "POST", "path": "/x", "headers": []}
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"z" * 500, "more_body": False}

    async def app(scope: dict, receive_: object, send_: object) -> None:
        await receive_()  # type: ignore[operator]  # trips the cap → disconnect injected
        raise ValueError("parser blew up on the truncated body")

    async def send(message: dict) -> None:
        sent.append(message)

    middleware = _RequestSizeLimitMiddleware(app, max_bytes=100)
    await middleware(scope, receive, send)  # must NOT raise
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 413


async def test_app_error_when_not_rejecting_propagates() -> None:
    # A genuine error on an in-cap request is NOT swallowed — it propagates.
    scope = {"type": "http", "method": "POST", "path": "/x", "headers": []}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"ok", "more_body": False}

    async def app(scope: dict, receive_: object, send_: object) -> None:
        raise ValueError("a real bug, unrelated to the size cap")

    async def send(message: dict) -> None:
        pass

    middleware = _RequestSizeLimitMiddleware(app, max_bytes=100)
    try:
        await middleware(scope, receive, send)
    except ValueError:
        pass
    else:  # pragma: no cover - the assert below reports the failure
        raise AssertionError("expected the genuine error to propagate")


async def test_honest_over_cap_content_length_short_circuits() -> None:
    # The fast path refuses a declared over-cap length before the app is invoked.
    status, inner_ran, count = await _drive(
        max_bytes=100,
        headers=[(b"content-length", b"500")],
        body_frames=[_frame(b"z" * 500, more=False)],
    )
    assert status == 413
    assert inner_ran is False
    assert count == 1


async def test_unparseable_content_length_falls_through_to_streaming_cap() -> None:
    # A malformed Content-Length does not short-circuit; the streaming cap catches
    # an over-cap body instead.
    status, inner_ran, count = await _drive(
        max_bytes=100,
        headers=[(b"content-length", b"not-a-number")],
        body_frames=[_frame(b"z" * 500, more=False)],
    )
    assert status == 413
    assert inner_ran is False
    assert count == 1
