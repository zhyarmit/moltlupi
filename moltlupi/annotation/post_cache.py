import json
import sqlite3
import time
from tqdm import tqdm
from urllib import request, error
from .annotate import encoded, digest, validate_response, ENDPOINT, NoRedirect

class Cache:

    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.executescript('\n        CREATE TABLE IF NOT EXISTS pc_posts (post_key TEXT PRIMARY KEY, original_json TEXT NOT NULL);\n        CREATE TABLE IF NOT EXISTS pc_occurrences (case_id TEXT, record_id TEXT, post_key TEXT, PRIMARY KEY(case_id,record_id));\n        CREATE TABLE IF NOT EXISTS pc_answers (cache_key TEXT PRIMARY KEY, answer_json TEXT NOT NULL, request_hash TEXT NOT NULL);\n        CREATE TABLE IF NOT EXISTS pc_artifact_annotations (post_key TEXT, config_hash TEXT, artifact_id TEXT, exact_text TEXT, normalized_key TEXT, role TEXT, annotation_json TEXT, PRIMARY KEY(post_key,config_hash,artifact_id));\n        CREATE TABLE IF NOT EXISTS pc_dimension_annotations (post_key TEXT, config_hash TEXT, dimension TEXT, label TEXT, annotation_json TEXT, PRIMARY KEY(post_key,config_hash,dimension));\n        CREATE TABLE IF NOT EXISTS pc_results (post_key TEXT, config_hash TEXT, result_json TEXT NOT NULL, PRIMARY KEY(post_key,config_hash));\n        CREATE TABLE IF NOT EXISTS pc_requests (request_hash TEXT PRIMARY KEY, request_json TEXT NOT NULL, response_json TEXT NOT NULL);\n        CREATE TABLE IF NOT EXISTS pc_attempts (id INTEGER PRIMARY KEY, request_hash TEXT, raw_response BLOB, error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);\n        ')
        self.db.commit()

    def get(self, key):
        row = self.db.execute('SELECT answer_json FROM pc_answers WHERE cache_key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.db.close()

class SQLiteJev:

    def __init__(self, cache, key=None, timeout=120, transport=None):
        self.cache = cache
        self.key = key
        self.timeout = timeout
        self.transport = transport
        self.network_calls = 0
        self.request_cache_hits = 0
        self.usage = []
        self.opener = request.build_opener(NoRedirect)

    def call(self, payload):
        rid = digest(dict(endpoint=ENDPOINT, payload=payload))
        row = self.cache.db.execute('SELECT response_json FROM pc_requests WHERE request_hash=?', (rid,)).fetchone()
        if row:
            try:
                checked = validate_response(json.loads(row[0]), payload)
            except ValueError:
                self.cache.db.execute('DELETE FROM pc_requests WHERE request_hash=?', (rid,))
                self.cache.db.commit()
            else:
                self.request_cache_hits += 1
                return checked
        if not self.transport and (not self.key):
            raise ValueError('Set TYPESAFE_API_KEY for live Jev calls')
        for attempt in range(4):
            raw = None
            try:
                self.network_calls += 1
                if self.transport:
                    raw = self.transport(payload)
                else:
                    req = request.Request(ENDPOINT, data=encoded(payload), method='POST', headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
                    with self.opener.open(req, timeout=self.timeout) as response:
                        raw = response.read()
                if isinstance(raw, dict):
                    raw = encoded(raw)
                self.cache.db.execute('INSERT INTO pc_attempts(request_hash,raw_response) VALUES (?,?)', (rid, raw))
                self.cache.db.commit()
                checked = validate_response(json.loads(raw), payload)
                with self.cache.db:
                    self.cache.db.execute('INSERT OR REPLACE INTO pc_requests VALUES (?,?,?)', (rid, encoded(payload).decode(), raw.decode()))
                self.usage.append(checked.get('usage', {}))
                return checked
            except (error.URLError, TimeoutError, ConnectionError) as exc:
                self.cache.db.execute('INSERT INTO pc_attempts(request_hash,error) VALUES (?,?)', (rid, type(exc).__name__ + ': ' + str(exc)))
                self.cache.db.commit()
                if isinstance(exc, error.HTTPError) and exc.code not in {429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}:
                    raise RuntimeError(f'Jev HTTP {exc.code}') from None
                if attempt == 3:
                    raise RuntimeError(f'Jev transport failed after 4 attempts: {type(exc).__name__}: {exc}. Resume with the same SQLite database and a new output directory; completed answers remain cached.') from None
                delay = 2 ** attempt
                if isinstance(exc, error.HTTPError):
                    try:
                        delay = max(delay, float((exc.headers or {}).get('Retry-After', 0)))
                    except (ValueError, TypeError):
                        pass
                if delay > 60:
                    raise RuntimeError('Rate limited for over 60 seconds; resume with same SQLite database') from None
                time.sleep(delay)
            except (ValueError, TypeError) as exc:
                self.cache.db.execute('INSERT INTO pc_attempts(request_hash,error) VALUES (?,?)', (rid, 'validation: ' + str(exc)))
                self.cache.db.commit()
                raise

def payload_for(tasks, model):
    states = {}
    for task in tasks:
        unit = task['unit_key']
        if unit in states and states[unit] != task['state']:
            raise ValueError('Conflicting states for unit ' + unit + '; refusing to overwrite model evidence')
        states[unit] = task['state']
    return dict(model=model, state={'units': states}, questions={t['question_key']: t['question'] for t in tasks})

def fits(tasks, model, state_budget, request_budget):
    if not tasks:
        return True
    p = payload_for(tasks, model)
    return len(encoded(p['state'])) + max((len(encoded(q)) for q in p['questions'].values())) <= state_budget and len(encoded(p)) <= request_budget

def pack(tasks, model, max_units=4, state_budget=28000, request_budget=58000):
    batch = []
    for task in tasks:
        trial = batch + [task]
        long_unit = any((t['state'].get('coverage', {}).get('source_truncated') or len(encoded(t['state'])) > state_budget // 2 for t in trial))
        if batch and (len({t['unit_key'] for t in trial}) > (1 if long_unit else max_units) or not fits(trial, model, state_budget, request_budget)):
            yield (payload_for(batch, model), batch)
            batch = []
        if not fits([task], model, state_budget, request_budget):
            single = payload_for([task], model)
            state_bytes = len(encoded(single['state']))
            question_bytes = max((len(encoded(q)) for q in single['questions'].values()), default=0)
            request_bytes = len(encoded(single))
            raise ValueError(f"One source/question exceeds configured request guards: unit={task['unit_key']}, state={state_bytes} bytes, question={question_bytes} bytes, request={request_bytes} bytes; limits state={state_budget}, request={request_budget}. Reduce the source excerpt or increase the guards.")
        batch.append(task)
    if batch:
        yield (payload_for(batch, model), batch)

def execute(tasks, cache, client, model, max_units, state_budget, request_budget, dry_run=False, progress=False, stage='Jev annotation'):
    answers = {}
    pending = []
    hits = 0
    for task in tqdm(tasks, desc=stage + ' — cache check', unit='question', disable=not progress, leave=False):
        a = cache.get(task['cache_key'])
        if a is not None:
            try:
                validate_response({'model': model, 'answers': {task['question_key']: a}}, payload_for([task], model))
            except ValueError:
                a = None
        if a is None:
            pending.append(task)
        else:
            answers[task['cache_key']] = a
            hits += 1
    report = {'questions': len(tasks), 'cached_questions': hits, 'pending_questions': len(pending), 'requests': 0, 'request_bytes': 0, 'max_units_per_request': 0}
    with tqdm(total=len(tasks), initial=hits, desc=stage + (' — preview' if dry_run else ''), unit='question', disable=not progress, dynamic_ncols=True) as bar:
        bar.set_postfix(cached=hits, batches=0, refresh=False)
        for payload, group in pack(pending, model, max_units, state_budget, request_budget):
            report['requests'] += 1
            report['request_bytes'] += len(encoded(payload))
            report['max_units_per_request'] = max(report['max_units_per_request'], len(payload['state']['units']))
            if dry_run:
                bar.update(len(group))
                bar.set_postfix(cached=hits, planned_batches=report['requests'], refresh=False)
                continue
            response = client.call(payload)
            rid = digest(dict(endpoint=ENDPOINT, payload=payload))
            with cache.db:
                for task in group:
                    answer = response['answers'][task['question_key']]
                    cache.db.execute('INSERT OR REPLACE INTO pc_answers VALUES (?,?,?)', (task['cache_key'], encoded(answer).decode(), rid))
                    answers[task['cache_key']] = answer
            bar.update(len(group))
            bar.set_postfix(cached=hits, batches=report['requests'], refresh=False)
    return (answers, report)
