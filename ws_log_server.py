import asyncio
import json
import logging
import threading
import os
from typing import Set, Optional

try:
    import websockets
except Exception:
    websockets = None


class WaldoLogWebSocketHub:
    def __init__(self):
        self.clients: Set["websockets.WebSocketServerProtocol"] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.server = None
        self.thread: Optional[threading.Thread] = None
        self.port = int(os.environ.get("LOG_WS_PORT", "5010"))

    async def _handler(self, websocket):
        self.clients.add(websocket)
        try:
            # Keep connection open; ignore incoming messages
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    async def _broadcast(self, message: str):
        if not self.clients:
            return
        coros = []
        for ws in list(self.clients):
            try:
                coros.append(ws.send(message))
            except Exception:
                self.clients.discard(ws)
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    def broadcast(self, payload: dict):
        if not self.loop:
            return
        try:
            msg = json.dumps(payload, ensure_ascii=False)
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self.loop)
        except Exception:
            pass

    def start(self):
        if websockets is None:
            logging.warning("websockets package not available; raw WS logging disabled")
            return False
        if self.thread and self.thread.is_alive():
            return True

        def runner():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.server = self.loop.run_until_complete(
                websockets.serve(self._handler, "0.0.0.0", self.port, ping_interval=25, ping_timeout=60)
            )
            logging.info(f"Waldo raw WebSocket log server listening on ws://0.0.0.0:{self.port}")
            try:
                self.loop.run_forever()
            finally:
                self.server.close()
                self.loop.run_until_complete(self.server.wait_closed())
                self.loop.close()

        self.thread = threading.Thread(target=runner, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        try:
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass

hub = WaldoLogWebSocketHub()

