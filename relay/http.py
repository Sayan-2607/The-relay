"""Minimal async HTTP/1.1 POST client using only the stdlib (TLS, chunked, SSE-friendly).
One connection per request (Connection: close). That adds a TLS handshake to every real call;
say so in your write-up, it slightly inflates real-service latency vs a keep-alive client."""
import asyncio, ssl as _ssl
from urllib.parse import urlsplit


class HttpResponse:
    def __init__(self, status, headers, reader, writer):
        self.status, self.headers, self.reader, self.writer = status, headers, reader, writer

    async def iter_body(self):
        r, te = self.reader, self.headers.get("transfer-encoding", "").lower()
        try:
            if "chunked" in te:
                while True:
                    line = await r.readline()
                    if not line:
                        raise ConnectionError("eof in chunk header")
                    size = int(line.split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        await r.readline()
                        break
                    data = await r.readexactly(size)
                    await r.readexactly(2)
                    yield data
            elif "content-length" in self.headers:
                remaining = int(self.headers["content-length"])
                while remaining > 0:
                    data = await r.read(min(65536, remaining))
                    if not data:
                        raise ConnectionError("eof before content-length")
                    remaining -= len(data)
                    yield data
            else:
                while True:
                    data = await r.read(65536)
                    if not data:
                        break
                    yield data
        except asyncio.IncompleteReadError as e:
            raise ConnectionError("incomplete read") from e

    async def read_all(self) -> bytes:
        return b"".join([c async for c in self.iter_body()])

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass


async def post(url: str, headers: dict, body: bytes) -> HttpResponse:
    u = urlsplit(url)
    tls = u.scheme == "https"
    port = u.port or (443 if tls else 80)
    kw = dict(ssl=_ssl.create_default_context(), server_hostname=u.hostname) if tls else {}
    reader, writer = await asyncio.open_connection(u.hostname, port, **kw)
    path = u.path + (("?" + u.query) if u.query else "")
    h = {"Host": u.netloc, "Content-Length": str(len(body)), "Connection": "close",
         "User-Agent": "relay-assignment/1.0", "Accept": "*/*", **headers}
    head = f"POST {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n"
    writer.write(head.encode() + body)
    await writer.drain()
    status_line = await reader.readline()
    if not status_line:
        writer.close()
        raise ConnectionError("empty response")
    status = int(status_line.split()[1])
    hdrs = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("latin-1").partition(":")
        hdrs[k.strip().lower()] = v.strip()
    return HttpResponse(status, hdrs, reader, writer)
