import re
from urllib.parse import urlsplit
from .annotate import encoded, digest, read_jsonl, write_json, choice_diagnostic, safe_cell, DEFAULT_MODEL
from .post_snapshots import candidates
from .post_level_rubric import VERSION, POLICY, DIMENSIONS, OBS, ARTIFACT_ROLES, RELATIONS, REL_CRITERIA, CLUSTER_OPTIONS

def normalized(value):
    value = value.strip()
    if re.match('^(?:https?://|www\\.)', value, re.I):
        try:
            u = urlsplit(value if '://' in value else 'https://' + value)
        except ValueError:
            return 'invalid_url:' + value
        return 'url:' + u.netloc.lower() + (u.path or '/') + ('?' + u.query if u.query else '') + ('#' + u.fragment if u.fragment else '')
    return 'name:' + re.sub('\\s+', ' ', value).casefold()

def extract_artifacts(text, terms=()):
    raw, _ = candidates(text)
    spans = {(v['start'], v['end']) for v in raw.values()}
    for term in terms:
        if not term:
            continue
        for m in re.finditer(re.escape(term), text, re.I):
            spans.add(m.span())
    rows = {}
    for start, end in sorted(spans):
        value = text[start:end]
        if re.match('^(?:https?://|www\\.)', value, re.I):
            while value and value[-1] in '.,;!':
                end -= 1
                value = text[start:end]
            while value.endswith(')') and value.count(')') > value.count('('):
                end -= 1
                value = text[start:end]
        if not value:
            continue
        key = normalized(value)
        if key not in rows:
            rows[key] = dict(text=value, start=start, end=end, normalized=key)
    result = sorted(rows.values(), key=lambda x: (0 if x['normalized'].startswith('url:') else 1 if '/' in x['text'] or x['text'].startswith('@') else 2, x['start']))
    return result

def segment(text):
    parts = []
    start = 0
    for m in re.finditer('(?<=[.!?。！？])\\s+|\\n+', text):
        if m.end() - start >= 100 or (m.end() - start > 0 and len(parts) < 200):
            parts.append((start, m.end()))
            start = m.end()
    if start < len(text):
        parts.append((start, len(text)))
    while len(parts) > 240:
        parts = [(parts[i][0], parts[min(i + 1, len(parts) - 1)][1]) for i in range(0, len(parts), 2)]
    return {f's{i:03d}': dict(start=a, end=b, text=text[a:b]) for i, (a, b) in enumerate(parts)}

def prepare_post(record, args):
    text = record['text']
    all_arts = extract_artifacts(text, args.artifact_term)

    def build(n):
        prefix = text[:n]
        arts = [a for a in all_arts if a['end'] <= n]
        return dict(author=record['author'], timestamp=record.get('timestamp'), source_segments=segment(prefix), artifacts={f'a{i:03d}': v for i, v in enumerate(arts[:args.max_artifacts])}, coverage=dict(original_chars=len(text), retained_chars=n, source_truncated=n < len(text), original_candidate_count=len(all_arts), candidates_outside_retained_text=len(all_arts) - len(arts), candidate_count=len(arts), candidate_limit=args.max_artifacts, candidates_omitted=max(0, len(arts) - args.max_artifacts)))
    result = build(len(text))
    limit = min(args.post_bytes, args.state_bytes - 5000)
    if len(encoded(result)) > limit:
        lo, hi = (0, len(text))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(encoded(build(mid))) <= limit:
                lo = mid
            else:
                hi = mid - 1
        if lo < 1:
            raise ValueError('Metadata/candidates leave no room for original post text')
        result = build(lo)
    return result

def task(unit, state, name, question, model):
    return dict(unit_key=unit, state=state, question_key=unit + '__' + name, question=question, cache_key=digest(dict(version=VERSION, model=model, unit=unit, state=state, name=name, question=question)))

def accepted(answer, args):
    return choice_diagnostic(answer) is None and answer['confidence'] >= args.min_confidence and (answer['probabilities'][answer['choice']] >= args.min_probability)

def citation_task(key, state, name, answer, args):
    unit = 'p' + key[:24]
    criteria = {'NONE': 'No source passage supports this interpretation.'}
    criteria.update({sid: 'Exact source segment ' + sid for sid in state['source_segments']})
    q = dict(type='choice', instructions=POLICY + '\nTarget units.' + unit + '. Select the source segment supporting ' + name + ' = ' + answer['choice'] + '. Inspect qualifiers and attribution; NONE if unsupported.', criteria=criteria)
    return task(unit, state, 'cite_' + name, q, args.model)
