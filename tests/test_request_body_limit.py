import asyncio

from hear.api.body_limit import RequestBodyLimitMiddleware


def test_body_limit_hands_receive_back_to_streaming_response():
    received = []
    sent = []

    async def app(scope, receive, send):
        received.append(await receive())
        received.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages = [
        {"type": "http.request", "body": b"{}", "more_body": False},
        {"type": "http.disconnect"},
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(
        RequestBodyLimitMiddleware(app, 16)(
            {
                "type": "http",
                "path": "/v1/attempts/stream",
                "headers": [(b"content-length", b"2")],
            },
            receive,
            send,
        )
    )

    assert received == [
        {"type": "http.request", "body": b"{}", "more_body": False},
        {"type": "http.disconnect"},
    ]
    assert sent[-1]["body"] == b"ok"


def test_body_limit_rejects_oversized_request_before_application():
    called = False
    sent = []

    async def app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(
        RequestBodyLimitMiddleware(app, 2)(
            {
                "type": "http",
                "path": "/v1/attempts/stream",
                "headers": [(b"content-length", b"3")],
            },
            receive,
            send,
        )
    )

    assert called is False
    assert sent[0]["status"] == 413
