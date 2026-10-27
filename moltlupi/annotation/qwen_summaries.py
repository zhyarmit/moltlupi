from .annotate import digest, encoded, write_json, parse_time, require_string, NoRedirect, PacketSizeError, preflight_requests, questions, DEFAULT_MODEL, ask, evidence_questions, finalize

class SummaryError(ValueError):
    pass

def check_hash(case):
    if case.get('packet_hash') and digest({k: v for k, v in case.items() if k != 'packet_hash'}) != case['packet_hash']:
        raise SummaryError('Packet hash mismatch')

def raw_state(case):
    check_hash(case)
    require_string(case.get('case_id'), 'case_id')
    members = case.get('members', [])
    if not isinstance(members, list) or len(members) < 2 or len(set(members)) != len(members):
        raise SummaryError('Need at least two distinct members')
    for member in members:
        require_string(member, 'member')
    scope = case.get('scope', {})
    start, end = (parse_time(scope.get('start')), parse_time(scope.get('end')))
    if scope.get('unit_type') not in {'pair_window', 'group_window'} or scope.get('boundary') != '[start,end)' or start is None or (end is None) or (start >= end):
        raise SummaryError('Require a bounded shared window with [start,end) timestamps')
    if scope['unit_type'] == 'pair_window' and len(members) != 2:
        raise SummaryError('Pair scope needs exactly two members')
    evidence, timestamps, ids, post_authors = ({}, [], set(), set())
    for r in case.get('records', []):
        rid = require_string(r.get('record_id'), 'record_id')
        text = require_string(r.get('text'), 'record text')
        if rid in ids:
            raise SummaryError('Duplicate record_id')
        ids.add(rid)
        if r.get('kind') not in {'post', 'profile'} or r.get('author') not in members:
            raise SummaryError('Only listed members posts/profiles are supported')
        if r.get('text_truncated') or r.get('original_text_chars', len(text)) != len(text):
            raise SummaryError('Summaries require full originals; rebuild truncated packets from the raw index')
        if r['kind'] == 'post':
            timestamp = parse_time(r.get('timestamp'))
            if timestamp is None or not start <= timestamp < end:
                raise SummaryError('Post missing a timestamp or outside the original shared window')
            post_authors.add(r['author'])
            timestamps.append((timestamp, rid, r['author']))
        eid = f'e{len(evidence):04d}'
        evidence[eid] = {k: r.get(k) for k in ('record_id', 'author', 'kind', 'timestamp')}
        evidence[eid].update(text=text, source_field='text', start=0, end=len(text), representation='full_source')
    if post_authors != set(members):
        raise SummaryError('Every member needs a sampled in-window post')
    timestamps.sort()
    origin = timestamps[0][0]
    return dict(members=members, scope=scope, evidence=evidence, records_without_timestamp=sum((r.get('timestamp') is None for r in case['records'])), timeline_computed_in_python=[dict(record_id=rid, author=a, utc=t.isoformat(), seconds_from_first_known_record=(t - origin).total_seconds()) for t, rid, a in timestamps])
