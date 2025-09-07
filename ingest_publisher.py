import os
import json
import time
import threading
import queue
import logging

try:
    import websockets
except Exception:
    websockets = None


class IngestPublisher:
    def __init__(self):
        self.url = os.environ.get('INGEST_WS_URL')
        self.enabled = os.environ.get('INGEST_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.q: "queue.Queue[bytes]" = queue.Queue(maxsize=100)
        self.thread: threading.Thread | None = None
        self.stop_flag = False
        self.width = None
        self.height = None

    def set_dims(self, w: int, h: int):
        self.width = int(w) if w else None
        self.height = int(h) if h else None

    def start(self):
        if not self.enabled or not self.url or websockets is None:
            logging.info("IngestPublisher disabled or websockets missing; skipping start")
            return False
        if self.thread and self.thread.is_alive():
            return True

        def worker():
            while not self.stop_flag:
                try:
                    import asyncio

                    async def run():
                        async with websockets.connect(self.url, max_size=None) as ws:
                            # Send optional hello message
                            try:
                                hello = {"type": "hello", "w": self.width, "h": self.height, "fmt": "jpeg"}
                                await ws.send(json.dumps(hello))
                            except Exception:
                                pass
                            # Streaming loop
                            while not self.stop_flag:
                                try:
                                    frame = self.q.get(timeout=0.5)
                                except queue.Empty:
                                    await asyncio.sleep(0.05)
                                    continue
                                try:
                                    await ws.send(frame)
                                except Exception as se:
                                    logging.warning(f"Ingest send error: {se}")
                                    break

                    asyncio.run(run())
                except Exception as e:
                    logging.warning(f"Ingest connection error; retrying soon: {e}")
                    time.sleep(2)

        self.stop_flag = False
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        logging.info("IngestPublisher started")
        return True

    def stop(self):
        self.stop_flag = True
        try:
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)
        except Exception:
            pass
        logging.info("IngestPublisher stopped")

    def enqueue(self, jpeg_bytes: bytes):
        if not self.enabled or not self.url or websockets is None:
            return
        try:
            self.q.put_nowait(jpeg_bytes)
        except queue.Full:
            # Drop oldest to avoid backpressure
            try:
                _ = self.q.get_nowait()
            except Exception:
                pass
            try:
                self.q.put_nowait(jpeg_bytes)
            except Exception:
                pass


publisher = IngestPublisher()

