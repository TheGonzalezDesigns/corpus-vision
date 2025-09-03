import os
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class EventStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get('VISION_EVENT_LOG', '/home/nerostar/Projects/corpus/vision_events.jsonl')
        self._lock = threading.Lock()
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except Exception:
            pass

    def append(self, event: Dict[str, Any]):
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

    def _read_all(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not os.path.exists(self.path):
            return events
        with self._lock:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        continue
        return events

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        events = self._read_all()
        return events[-limit:] if limit and len(events) > limit else events

    def range(self, from_iso: str, to_iso: str) -> List[Dict[str, Any]]:
        events = self._read_all()
        try:
            start = datetime.fromisoformat(from_iso.replace('Z', '+00:00'))
            end = datetime.fromisoformat(to_iso.replace('Z', '+00:00'))
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for ev in events:
            ts = ev.get('ts_iso') or ev.get('ts')
            try:
                dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
            except Exception:
                continue
            if start <= dt <= end:
                out.append(ev)
        return out

    def context(self, window_minutes: int = 15, limit: int = 20) -> Dict[str, Any]:
        events = self._read_all()
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=window_minutes)
        selected: List[Dict[str, Any]] = []
        for ev in reversed(events):
            ts = ev.get('ts_iso') or ev.get('ts')
            try:
                dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
            except Exception:
                continue
            if dt >= cutoff:
                # Keep only essentials for prompt context
                selected.append({
                    'ts_iso': ev.get('ts_iso'),
                    'duration_ms': ev.get('duration_ms'),
                    'frames_count': ev.get('frames_count'),
                    'description': ev.get('description'),
                    'confidence_hint': ev.get('confidence_hint'),
                })
                if len(selected) >= limit:
                    break
        selected.reverse()
        return {
            'window_minutes': window_minutes,
            'limit': limit,
            'count': len(selected),
            'events': selected
        }


# Shared store instance for easy reuse
store = EventStore()

