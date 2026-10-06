"""Mimiq Link — turns an iPhone (Safari, no app needed) into a Wi-Fi webcam.

The PC serves a small web app over HTTPS. The phone captures its camera with
getUserMedia, encodes JPEG frames and sends them over a WebSocket. Each frame is
acknowledged, and the phone keeps at most two frames in flight, so latency stays
low and nothing queues up when Wi-Fi gets busy.

Binary frame layout (little endian):  b"MQ" | u8 version | u8 flags | u32 seq | u32 t_ms | JPEG…
"""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
import struct
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .. import paths
from .certs import ensure_certificates, local_ipv4s
from .sources import LinkSource

log = logging.getLogger("mimiq.link")

HEADER = struct.Struct("<2sBBII")


class LinkServer:
    def __init__(self, port: int, token: str, on_event: Optional[Callable[[str, Dict], None]] = None):
        self.port = port
        self.token = token
        self.on_event = on_event
        self.sink: Optional[LinkSource] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner = None
        self._ws = None
        self._ready = threading.Event()
        self.error: Optional[str] = None
        self.client: Dict[str, object] = {}
        self.ips: List[str] = []
        self.stats = {"fps": 0.0, "kbps": 0.0, "rtt": 0.0}
        self._bytes = 0
        self._frames = 0
        self._stat_t = time.monotonic()
        self.web_root = paths.package_dir() / "web"
        # HTTPS fallback transport (Safari may refuse WSS with a self-signed certificate)
        self._http_active = False
        self._http_seen = 0.0
        self._outbox: List[Dict] = []
        self._wd_task = None

    # ------------------------------------------------------------------
    def urls(self) -> List[str]:
        return [f"https://{ip}:{self.port}/?t={self.token}" for ip in self.ips]

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self.error is None

    @property
    def connected(self) -> bool:
        return self._ws is not None or self._http_active

    def start(self) -> bool:
        if self.running:
            return True
        self.error = None
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, name="mimiq-link", daemon=True)
        self._thread.start()
        self._ready.wait(8)
        return self.error is None

    def stop(self) -> None:
        if self.loop and self.loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
            try:
                fut.result(timeout=3)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        self._ws = None

    def send(self, message: Dict) -> None:
        """Send a control message (camera, resolution, quality…) to the phone."""
        if self.loop and self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._send(message), self.loop)
        elif self.loop and self._http_active:
            def queue():
                if message.get("t") == "config":
                    self._outbox[:] = [m for m in self._outbox if m.get("t") != "config"]
                self._outbox.append(message)
            self.loop.call_soon_threadsafe(queue)

    # ------------------------------------------------------------------
    def _emit(self, kind: str, data: Optional[Dict] = None) -> None:
        if self.on_event:
            try:
                self.on_event(kind, data or {})
            except Exception:
                log.exception("link event handler failed")

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._startup())
        except Exception as exc:
            self.error = f"Не удалось запустить Mimiq Link на порту {self.port}: {exc}"
            log.error(self.error)
            self._ready.set()
            self._emit("error", {"message": self.error})
            return
        self._ready.set()
        self._emit("listening", {"urls": self.urls(), "ips": self.ips})
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    async def _startup(self) -> None:
        from aiohttp import web
        self.ips = local_ipv4s() or ["127.0.0.1"]
        chain, key, ca = ensure_certificates(paths.certs_dir(), self.ips)
        self.ca_path = ca
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        try:
            ctx.load_cert_chain(str(chain), str(key))
        except (OSError, ssl.SSLError):
            # very old OpenSSL builds can't open non-ASCII (e.g. Cyrillic user name) paths on Windows
            import shutil
            from ..imaging import ascii_safe_dir
            tmp = ascii_safe_dir(chain.parent)
            shutil.copyfile(chain, tmp / "mimiq-chain.pem")
            shutil.copyfile(key, tmp / "mimiq-key.pem")
            try:
                ctx.load_cert_chain(str(tmp / "mimiq-chain.pem"), str(tmp / "mimiq-key.pem"))
            finally:
                for name in ("mimiq-chain.pem", "mimiq-key.pem"):
                    if tmp != chain.parent:
                        (tmp / name).unlink(missing_ok=True)
        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._websocket)
        app.router.add_get("/ca.crt", self._ca)
        app.router.add_get("/health", self._health)
        app.router.add_post("/frame", self._http_frame)
        app.router.add_post("/hello", self._http_hello)
        app.router.add_post("/stats", self._http_stats)
        app.router.add_static("/static/", self.web_root, show_index=False)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        last_exc: Optional[Exception] = None
        for port in range(self.port, self.port + 10):   # the preferred port may be taken by another app
            site = web.TCPSite(self._runner, "0.0.0.0", port, ssl_context=ctx)
            try:
                await site.start()
            except OSError as exc:
                last_exc = exc
                log.warning("port %d is busy (%s), trying the next one", port, exc)
                continue
            self.port = port
            break
        else:
            raise last_exc or OSError("no free port")
        self._wd_task = asyncio.ensure_future(self._watchdog())
        log.info("Mimiq Link listening on %s", ", ".join(self.urls()))

    async def _shutdown(self) -> None:
        if self._wd_task is not None:
            self._wd_task.cancel()
        if self._ws is not None:
            await self._ws.close()
        if self._runner is not None:
            await self._runner.cleanup()

    async def _send(self, message: Dict) -> None:
        ws = self._ws
        if ws is not None and not ws.closed:
            await ws.send_str(json.dumps(message))

    # ------------------------------------------------------------------
    async def _health(self, request):
        from aiohttp import web
        return web.json_response({"ok": True, "app": "Mimiq"})

    async def _index(self, request):
        from aiohttp import web
        html = (self.web_root / "index.html").read_text(encoding="utf-8")
        authorised = request.query.get("t") == self.token
        html = html.replace("__MIMIQ_AUTH__", "true" if authorised else "false")
        return web.Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def _ca(self, request):
        from aiohttp import web
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import Encoding
        der = x509.load_pem_x509_certificate(Path(self.ca_path).read_bytes()).public_bytes(Encoding.DER)
        return web.Response(body=der, content_type="application/x-x509-ca-cert",
                            headers={"Content-Disposition": 'attachment; filename="Mimiq-Local-CA.cer"',
                                     "Cache-Control": "no-store"})

    async def _websocket(self, request):
        from aiohttp import web, WSMsgType
        if request.query.get("t") != self.token:
            raise web.HTTPForbidden(text="bad token")
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=5.0, compress=False)
        await ws.prepare(request)
        if self._ws is not None and not self._ws.closed:  # newest phone wins
            await self._ws.close(message=b"replaced")
        self._ws = ws
        self._http_active = False
        peer = request.remote or "?"
        self.client = {"ip": peer, "transport": "wss"}
        self._emit("connected", dict(self.client))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    self._on_frame(msg.data, ws)
                elif msg.type == WSMsgType.TEXT:
                    await self._on_text(msg.data, ws)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            if self._ws is ws:
                self._ws = None
                if self.sink:
                    self.sink.disconnected()
                self._emit("disconnected", dict(self.client))
        return ws

    def _on_frame(self, data: bytes, ws) -> None:
        if len(data) < HEADER.size + 4:
            return
        magic, _ver, _flags, seq, t_ms = HEADER.unpack_from(data)
        if magic != b"MQ":
            return
        self._ingest(data[HEADER.size:], len(data))
        asyncio.ensure_future(ws.send_str(f'{{"t":"ack","s":{seq},"c":{t_ms}}}'))

    def _ingest(self, jpeg: bytes, nbytes: int) -> None:
        if self.sink is not None:
            self.sink.push_jpeg(jpeg)
        self._frames += 1
        self._bytes += nbytes
        now = time.monotonic()
        if now - self._stat_t >= 1.0:
            dt_ = now - self._stat_t
            self.stats.update(fps=self._frames / dt_, kbps=self._bytes * 8 / 1000 / dt_)
            self._frames, self._bytes, self._stat_t = 0, 0, now
            self._emit("stats", dict(self.stats))

    async def _on_text(self, text: str, ws) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            return
        self._handle_message(msg)

    def _handle_message(self, msg: Dict) -> None:
        kind = msg.get("t")
        if kind == "hello":
            self.client.update({k: msg.get(k) for k in ("device", "camera", "width", "height", "fps", "ua")})
            if self.sink:
                self.sink.connected({"device": msg.get("device") or "iPhone", "camera": msg.get("camera"),
                                     "width": msg.get("width"), "height": msg.get("height")})
            self._emit("hello", dict(self.client))
        elif kind == "stats":
            self.stats["rtt"] = float(msg.get("rtt") or 0)
            self.client.update({k: msg.get(k) for k in ("width", "height", "camera", "battery") if k in msg})
            self._emit("client_stats", dict(msg))

    # ------------------------------------------------------------------ HTTPS fallback transport
    def _authorise(self, request) -> None:
        from aiohttp import web
        if request.query.get("t") != self.token:
            raise web.HTTPForbidden(text="bad token")

    def _http_touch(self, request) -> None:
        self._http_seen = time.monotonic()
        if not self._http_active:
            self._http_active = True
            self.client = {"ip": request.remote or "?", "transport": "https"}
            self._emit("connected", dict(self.client))

    def _http_reply(self, extra: Optional[Dict] = None):
        from aiohttp import web
        out: Dict = {"ok": True}
        if extra:
            out.update(extra)
        if self._outbox:
            out["msgs"], self._outbox = self._outbox, []
        return web.json_response(out, headers={"Cache-Control": "no-store"})

    async def _http_frame(self, request):
        from aiohttp import web
        self._authorise(request)
        data = await request.read()
        self._http_touch(request)
        if len(data) < HEADER.size + 4:
            raise web.HTTPBadRequest(text="short frame")
        magic, _ver, _flags, seq, t_ms = HEADER.unpack_from(data)
        if magic != b"MQ":
            raise web.HTTPBadRequest(text="bad magic")
        self._ingest(data[HEADER.size:], len(data))
        return self._http_reply({"t": "ack", "s": seq, "c": t_ms})

    async def _http_hello(self, request):
        self._authorise(request)
        msg = await request.json()
        self._http_touch(request)
        msg["t"] = "hello"
        self._handle_message(msg)
        return self._http_reply()

    async def _http_stats(self, request):
        self._authorise(request)
        msg = await request.json()
        self._http_touch(request)
        msg["t"] = "stats"
        self._handle_message(msg)
        return self._http_reply()

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self._http_active and time.monotonic() - self._http_seen > 4.0:
                self._http_active = False
                if self.sink:
                    self.sink.disconnected()
                self._emit("disconnected", dict(self.client))
