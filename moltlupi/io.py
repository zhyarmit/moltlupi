import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def timestamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamps must include a timezone')
    return dt.astimezone(timezone.utc)


def identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('IDs must be nonempty strings')
    return value


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Expected a finite probability in [0,1]')
    return float(value)


def load_windows(path):
    result = {}
    for row in read_json(path):
        key = identifier(row['window_id'])
        if key in result or timestamp(row['start']) >= timestamp(row['end']):
            raise ValueError('Duplicate or invalid window')
        result[key] = row
    if not result:
        raise ValueError('No observation windows')
    return result
