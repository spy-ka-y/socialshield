"""Incident store: every alert SocialShield raises, and what was done about it.

Detect -> respond -> record. Alerts from the live feed, the DM inbox and the message scanner
are saved here; the analyst can Block the sender, Quarantine the message or mark a False alarm.
Stored as a small JSON file (data/incidents.json) so the log survives app restarts.
"""
import json
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/incidents.json"
_LOCK = threading.Lock()

OPEN = "Open"
ACTIONS = {"block": "Sender blocked", "quarantine": "Message quarantined", "dismiss": "False alarm"}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read():
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _write(items):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PATH)


def all_incidents():
    with _LOCK:
        return _read()


def add(key, source, alert, risk, text, sender="", detected_by="Hybrid detector"):
    """Log an alert once. `key` makes repeated scans of the same item a no-op."""
    with _LOCK:
        items = _read()
        if any(i["key"] == key for i in items):
            return False
        items.append({"id": len(items) + 1, "key": key, "time": _now(), "source": source, "alert": alert,
                      "risk": round(float(risk), 3), "text": text, "sender": sender, "detected_by": detected_by,
                      "status": OPEN, "action_time": ""})
        _write(items)
        return True


def add_many(rows):
    """rows: iterable of dicts with the add() fields. One file write for the whole batch."""
    with _LOCK:
        items = _read()
        keys = {i["key"] for i in items}
        n = 0
        for r in rows:
            if r["key"] in keys:
                continue
            keys.add(r["key"])
            items.append({"id": len(items) + 1, "time": _now(), "status": OPEN, "action_time": "",
                          "detected_by": "Hybrid detector", **r, "risk": round(float(r["risk"]), 3)})
            n += 1
        if n:
            _write(items)
        return n


def act(incident_id, action):
    with _LOCK:
        items = _read()
        for i in items:
            if i["id"] == incident_id:
                i["status"] = ACTIONS[action]
                i["action_time"] = _now()
        _write(items)


def reopen(incident_id):
    with _LOCK:
        items = _read()
        for i in items:
            if i["id"] == incident_id:
                i["status"], i["action_time"] = OPEN, ""
        _write(items)


def clear():
    with _LOCK:
        _write([])
