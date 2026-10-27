import argparse, copy, json, os
from pathlib import Path
from tqdm import tqdm
from .annotate import read_jsonl, write_json, encoded, digest, choice_diagnostic
from .qwen_summaries import raw_state
from .post_cache import Cache, SQLiteJev, execute, fits
from .post_cluster_pipeline import prepare_post, task, accepted, citation_task, extract_artifacts
from .post_level_rubric import DIMENSIONS, VERSION, POLICY, ARTIFACT_ROLES, INFLUENCE_QUESTIONS
from .remote_artifacts import RemoteArtifactExtractor, ArtifactCacheMiss
ABSENT = {'not_observed', 'not_stated', 'not_described', 'unclear'}

def ignored_artifact(item, args):

    def name(value):
        return ' '.join(value.strip().lstrip('#').split()).casefold()
    ignored = {'moltbook'} | {name(x) for x in getattr(args, 'ignore_artifact', [])}
    return name(item['text']) in ignored

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--sqlite', required=True)
    p.add_argument('--model', default='jev-1.13.0')
    p.add_argument('--batch-posts', type=int, default=4)
    p.add_argument('--state-bytes', type=int, default=28000)
    p.add_argument('--request-bytes', type=int, default=58000)
    p.add_argument('--post-bytes', type=int, default=16000)
    p.add_argument('--max-artifacts', type=int, default=12)
    p.add_argument('--artifact-extractor', choices=['remote', 'regex'], default='remote')
    p.add_argument('--ignore-artifact', action='append', default=[], help='Ignore an exact artifact name (repeatable, case insensitive). Moltbook is always ignored.')
    p.add_argument('--artifact-backend', choices=['ollama', 'openai'], default='ollama')
    p.add_argument('--artifact-cache-only', action='store_true', help='Use cached remote extractions only; defer cache misses without model requests.')
    p.add_argument('--retry-failed-artifacts', action='store_true', help='Explicitly retry posts recorded as failed in prior runs.')
    p.add_argument('--artifact-context-tokens', type=int, default=12288)
    p.add_argument('--artifact-base-url', default=os.environ.get('ARTIFACT_LLM_BASE_URL', 'http://127.0.0.1:11434'))
    p.add_argument('--artifact-model', default=os.environ.get('ARTIFACT_LLM_MODEL'))
    p.add_argument('--artifact-source-chars', type=int, default=30000)
    p.add_argument('--artifact-max-output-tokens', type=int, default=800)
    p.add_argument('--artifact-reasoning-effort', default='none', help='Reasoning effort for the OpenAI-compatible artifact endpoint; use none for Ollama Gemma to reserve output for JSON.')
    p.add_argument('--artifact-term', action='append', default=[])
    p.add_argument('--min-probability', type=float, default=0.6)
    p.add_argument('--min-confidence', type=float, default=0.6)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--reset-artifacts', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    return p

def setup_artifact_tables(cache):
    cache.db.executescript('\n    CREATE TABLE IF NOT EXISTS pc_remote_artifacts(cache_key TEXT PRIMARY KEY,post_key TEXT,result_json TEXT NOT NULL);\n    CREATE TABLE IF NOT EXISTS pc_artifact_history(id INTEGER PRIMARY KEY,post_key TEXT,payload_json TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);\n    CREATE TABLE IF NOT EXISTS pc_artifact_generation(post_key TEXT PRIMARY KEY,generation INTEGER NOT NULL);\n    CREATE TABLE IF NOT EXISTS pc_artifact_failures(post_key TEXT PRIMARY KEY,failure_json TEXT NOT NULL);\n    ')
    cache.db.commit()

def reset_artifacts(cache, keys):
    archived = 0
    with cache.db:
        for key in keys:
            for row in cache.db.execute('SELECT config_hash,artifact_id,annotation_json FROM pc_artifact_annotations WHERE post_key=?', (key,)):
                cache.db.execute('INSERT INTO pc_artifact_history(post_key,payload_json) VALUES (?,?)', (key, encoded(dict(config_hash=row[0], artifact_id=row[1], annotation=json.loads(row[2]))).decode()))
                archived += 1
            cache.db.execute('DELETE FROM pc_artifact_annotations WHERE post_key=?', (key,))
            cache.db.execute('DELETE FROM pc_remote_artifacts WHERE post_key=?', (key,))
            old = cache.db.execute('SELECT generation FROM pc_artifact_generation WHERE post_key=?', (key,)).fetchone()
            cache.db.execute('INSERT OR REPLACE INTO pc_artifact_generation VALUES (?,?)', (key, (old[0] if old else 0) + 1))
    return {'posts': len(keys), 'archived_rows': archived}

def source_artifact_tasks(key, source, extraction, generation, args):
    artifacts = {f'a{index:03d}': item for index, item in enumerate(extraction['items']) if not ignored_artifact(item, args)}
    unit = 'a' + key[:20]
    text = source['text'][:args.artifact_source_chars]

    def build(n, truncated=False):
        tasks = []
        state = {'schema_version': 'artifact-role-targets-v2', 'post_key': key, 'author': source['author'], 'artifact_generation': generation, 'artifacts': artifacts, 'source_text': text[:n]}
        if truncated:
            state['source_coverage'] = dict(original_chars=len(source['text']), retained_chars=n, source_truncated=True, selection='exact_prefix', interpretation='The tail is omitted. Artifact offsets refer to the original post. Use unclear when the retained text does not establish a role.')
        for aid, item in artifacts.items():
            question = {'type': 'choice', 'instructions': POLICY + f'\nTarget ONLY units.{unit}.artifacts.{aid}; exact text: ' + json.dumps(item['text'], ensure_ascii=False) + f'. Use units.{unit}.source_text to classify the role of that specific occurrence. Other listed artifacts are context, not the target. An API endpoint may be an operational resource; do not transfer that role to a named account or profile URL merely because API instructions appear nearby. Use unclear if the target role is not established. Relevance is not wrongdoing. Do not infer the role from other posts.', 'criteria': ARTIFACT_ROLES}
            t = task(unit, state, 'artifact_' + aid, question, args.model)
            t.update(post_key=key, artifact_id=aid, artifact=item, source_coverage=dict(original_chars=len(source['text']), retained_chars=n, source_truncated=n < len(source['text']), target_in_retained_text=item['end'] <= n))
            tasks.append(t)
        return tasks

    def within(tasks):
        return all((fits([t], args.model, getattr(args, 'state_bytes', 28000), getattr(args, 'request_bytes', 58000)) for t in tasks))
    tasks = build(len(text))
    if within(tasks):
        return tasks
    if not within(build(0, True)):
        raise ValueError(f'Artifact metadata/question for {key} exceeds request guards even without source text')
    lo, hi = (0, len(text))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if within(build(mid, True)):
            lo = mid
        else:
            hi = mid - 1
    if text and lo == 0:
        raise ValueError(f'No budget for artifact source text for {key}')
    return build(lo, True)

def prepare_extraction(cache, extractor, key, src, generation, args):
    previous = cache.db.execute('SELECT failure_json FROM pc_artifact_failures WHERE post_key=?', (key,)).fetchone()
    if previous and (not getattr(args, 'retry_failed_artifacts', False)):
        return (None, dict(json.loads(previous[0]), status='skipped_previous_failure'))
    try:
        extraction = extractor.extract(key, src['text'], generation) if extractor else dict(items=extract_artifacts(src['text'], args.artifact_term)[:args.max_artifacts], extractor={'version': 'regex'})
        tasks = source_artifact_tasks(key, src, extraction, generation, args)
    except ArtifactCacheMiss as exc:
        return (None, dict(post_key=key, source=src, status='skipped_uncached', error=str(exc)))
    except Exception as exc:
        failure = dict(post_key=key, source=src, status='failed', error_type=type(exc).__name__, error=str(exc))
        with cache.db:
            cache.db.execute('INSERT OR REPLACE INTO pc_artifact_failures VALUES (?,?)', (key, encoded(failure).decode()))
        return (None, failure)
    with cache.db:
        cache.db.execute('DELETE FROM pc_artifact_failures WHERE post_key=?', (key,))
    return ((extraction, tasks), None)

def noise_report(cases, case_keys, results):
    reports = []
    for case in cases:
        by_agent = {m: {'keys': set(), 'posts': [], 'limitations': []} for m in case['members']}
        for key in case_keys[case['case_id']]:
            r = results[key]
            entry = by_agent.setdefault(r['author'], {'keys': set(), 'posts': [], 'limitations': []})
            entry['posts'].append(key)
            if r['artifact_status'] != 'complete':
                entry['limitations'].append('artifact_extraction_failed')
                continue
            for d in r['dimensions'].values():
                if d.get('source_coverage', {}).get('source_truncated'):
                    entry['limitations'].append('dimension_source_truncated')
            objective = r['dimensions'].get('stated_objective', {}).get('answer', {}).get('choice')
            purpose = r['dimensions'].get('purpose', {}).get('answer', {}).get('choice')
            if purpose == 'criticism' or objective in ABSENT:
                entry['limitations'].append('no_clear_objective')
            for a in r['artifacts']:
                if a['accepted'] and a['answer']['choice'] in {'beneficiary', 'promoted_entity', 'action_destination'}:
                    entry['keys'].add(a['normalized'])
        owners = {k: {a for a, v in by_agent.items() if k in v['keys']} for a in by_agent for k in by_agent[a]['keys']}
        agents = []
        for agent, v in by_agent.items():
            shared = {k: sorted(owners[k] - {agent}) for k in v['keys'] if len(owners[k]) > 1}
            status = 'shared_objective_artifact' if shared else 'insufficient_evidence' if v['limitations'] or not v['keys'] else 'potential_noise'
            agents.append(dict(agent=agent, status=status, objective_artifact_keys=sorted(v['keys']), shared_with=shared, post_keys=v['posts'], limitations=sorted(set(v['limitations']))))
        reports.append(dict(case_id=case['case_id'], agents=agents, potential_noise_agents=[a['agent'] for a in agents if a['status'] == 'potential_noise'], insufficient_evidence_agents=[a['agent'] for a in agents if a['status'] == 'insufficient_evidence']))
    return dict(version='artifact-overlap-screen-v2', automatic_exclusion=False, limitation='Potential noise means no shared accepted artifact was found; it is not proof of unrelated intent. Review before excluding.', cases=reports)

def run(args, transport=None, artifact_transport=None):
    if getattr(args, 'artifact_cache_only', False) and args.artifact_extractor != 'remote':
        raise ValueError('--artifact-cache-only requires --artifact-extractor remote')
    if getattr(args, 'artifact_cache_only', False) and args.reset_artifacts:
        raise ValueError('Do not combine --artifact-cache-only with --reset-artifacts, which removes those caches')
    if args.artifact_extractor == 'remote' and (not args.artifact_model) and (not args.dry_run):
        raise ValueError('--artifact-model is required for remote extraction')
    cases = list(read_jsonl(args.input))
    cases = cases[:args.limit] if args.limit else cases
    if not cases:
        raise ValueError('No cases')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    cache = Cache(args.sqlite)
    setup_artifact_tables(cache)
    for c in cases:
        raw_state(c)
    client = SQLiteJev(cache, os.environ.get('TYPESAFE_API_KEY'), args.timeout, transport)
    extractor = RemoteArtifactExtractor(cache, args, artifact_transport) if args.artifact_extractor == 'remote' else None
    sources = {}
    case_keys = {}
    states = {}
    results = {}
    dimension_tasks = []
    metadata = {}
    progress = not args.no_progress
    try:
        for c in cases:
            keys = []
            for r in c['records']:
                if r['kind'] != 'post':
                    continue
                src = {k: r.get(k) for k in ('author', 'timestamp', 'text')}
                key = digest(src)
                sources[key] = src
                keys.append(key)
                with cache.db:
                    cache.db.execute('INSERT OR IGNORE INTO pc_posts VALUES (?,?)', (key, encoded(src).decode()))
                    cache.db.execute('INSERT OR REPLACE INTO pc_occurrences VALUES (?,?,?)', (c['case_id'], r['record_id'], key))
            case_keys[c['case_id']] = list(dict.fromkeys(keys))
        artifact_reset = None
        if args.reset_artifacts and (not args.dry_run):
            artifact_reset = reset_artifacts(cache, sources)
        generations = {k: (cache.db.execute('SELECT generation FROM pc_artifact_generation WHERE post_key=?', (k,)).fetchone() or (0,))[0] for k in sources}
        for key, src in tqdm(sources.items(), desc='Prepare posts', unit='post', disable=not progress):
            states[key] = prepare_post(src, args)
            results[key] = dict(post_key=key, author=src['author'], timestamp=src['timestamp'], dimensions={}, artifacts=[], artifact_status='pending')
            for dim, (question, criteria) in DIMENSIONS.items():
                row = cache.db.execute('SELECT annotation_json FROM pc_dimension_annotations WHERE post_key=? AND dimension=? ORDER BY rowid DESC', (key, dim)).fetchone()
                if row:
                    results[key]['dimensions'][dim] = json.loads(row[0])
                    continue
                unit = 'p' + key[:24]
                target = '\nTarget ONLY units.' + unit + '. ' if dim in INFLUENCE_QUESTIONS else '\nTarget only this post. '
                q = {'type': 'choice', 'instructions': POLICY + target + question, 'criteria': criteria}
                t = task(unit, states[key], 'dim_' + dim, q, args.model)
                dimension_tasks.append(t)
                metadata[t['cache_key']] = (key, dim)
        dimension_answers, dim_report = execute(dimension_tasks, cache, client, args.model, args.batch_posts, args.state_bytes, args.request_bytes, args.dry_run, progress=progress, stage='Post dimensions')
        report = {'version': 'post-artifacts-v2-no-relations', 'dimensions': dim_report, 'relationship_checks': False, 'cluster_assessment': False, 'artifact_extractor': args.artifact_extractor, 'status': 'planned' if args.dry_run else 'running'}
        if artifact_reset is not None:
            report['artifact_reset'] = artifact_reset
        if args.dry_run:
            write_json(out / 'run_report.json', report)
            return report
        for t in dimension_tasks:
            key, dim = metadata[t['cache_key']]
            a = dimension_answers[t['cache_key']]
            d = dict(answer=a, accepted=accepted(a, args), diagnostic=choice_diagnostic(a), source_coverage=states[key]['coverage'])
            results[key]['dimensions'][dim] = d
            with cache.db:
                cache.db.execute('INSERT OR REPLACE INTO pc_dimension_annotations VALUES (?,?,?,?,?)', (key, 'dimension-v2', dim, a['choice'], encoded(d).decode()))
        artifact_tasks = []
        extraction_failures = 0
        extraction_skipped = 0
        failure_path = out / 'artifact_failures.jsonl'
        failure_path.touch()
        extraction_progress = tqdm(sources.items(), desc='Extract artifacts', unit='post', disable=not progress)
        for key, src in extraction_progress:
            prepared, failure = prepare_extraction(cache, extractor, key, src, generations[key], args)
            if failure is None:
                extraction, tasks = prepared
                results[key]['ignored_artifacts'] = [dict(item, ignore_reason='ignored_artifact_name') for item in extraction['items'] if ignored_artifact(item, args)]
                results[key].update(artifact_status='complete', artifact_coverage=extraction.get('coverage', {}), artifact_extractor=extraction.get('extractor', {}))
                artifact_tasks.extend(tasks)
            else:
                extraction_failures += int(failure['status'] == 'failed')
                extraction_skipped += int(failure['status'] != 'failed')
                results[key].update(artifact_status=failure['status'], artifact_error=failure['error'])
                with failure_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + '\n')
                    handle.flush()
            if extractor:
                extraction_progress.set_postfix(cached=extractor.hits, requests=extractor.calls, failed=extraction_failures, skipped=extraction_skipped, refresh=False)
        report['artifact_extraction'] = dict(cached=extractor.hits if extractor else 0, requests=extractor.calls if extractor else 0, failed=extraction_failures, skipped=extraction_skipped)
        artifact_answers, art_report = execute(artifact_tasks, cache, client, args.model, args.batch_posts, args.state_bytes, args.request_bytes, False, progress=progress, stage='Artifact roles')
        report['artifact_roles'] = art_report
        for t in artifact_tasks:
            a = artifact_answers[t['cache_key']]
            entry = dict(t['artifact'], artifact_id=t['artifact_id'], source_coverage=t['source_coverage'], answer=a, accepted=accepted(a, args), diagnostic=choice_diagnostic(a))
            results[t['post_key']]['artifacts'].append(entry)
            with cache.db:
                cache.db.execute('INSERT OR REPLACE INTO pc_artifact_annotations VALUES (?,?,?,?,?,?,?)', (t['post_key'], 'artifact-v2', t['artifact_id'], entry['text'], entry['normalized'], a['choice'], encoded(entry).decode()))
        with (out / 'post_annotations.jsonl').open('w', encoding='utf-8') as f:
            for key, r in results.items():
                f.write(json.dumps(dict(r, source=sources[key]), ensure_ascii=False) + '\n')
        write_json(out / 'potential_noise_agents.json', noise_report(cases, case_keys, results))
        report['status'] = 'complete'
        write_json(out / 'run_report.json', report)
        return report
    finally:
        cache.close()
