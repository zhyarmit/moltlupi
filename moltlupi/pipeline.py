import json
from collections import Counter, defaultdict
from pathlib import Path

from .annotation.assess_clusters_rules import assess, digest
from .annotation.post_annotation_pipeline import parser as annotation_parser, run as annotate
from .annotation.qwen_summaries import raw_state
from .discovery import discover
from .io import identifier, load_windows, read_json, read_jsonl, sha256, timestamp, write_json, write_jsonl


PRIORITY = {'insufficient_evidence': 0, 'shared_objective': 1, 'propagation': 2}


def make_packets(clusters, windows, posts_path, out):
    by_author = defaultdict(list)
    seen = {}
    for post in read_jsonl(posts_path):
        rid = identifier(post['record_id'])
        identifier(post['author'])
        identifier(post['text'])
        timestamp(post['timestamp'])
        if rid in seen:
            if seen[rid] != post:
                raise ValueError('Conflicting post ID: ' + rid)
            continue
        seen[rid] = post
        by_author[post['author']].append(post)
    packets, excluded = [], []
    for cluster in clusters:
        if not cluster['window_ids']:
            excluded.append({'cluster_id': cluster['cluster_id'], 'reason': 'no_common_observation_window'})
        for w in cluster['window_ids']:
            window = windows[w]
            start, end = timestamp(window['start']), timestamp(window['end'])
            records = [dict(p, kind='post') for a in cluster['members'] for p in by_author[a] if start <= timestamp(p['timestamp']) < end]
            absent = sorted(set(cluster['members']) - {p['author'] for p in records})
            if absent:
                excluded.append({'cluster_id': cluster['cluster_id'], 'window_id': w, 'reason': 'members_without_posts', 'members': absent})
                continue
            packet = {'case_id': cluster['cluster_id'] + ':' + w, 'cluster_id': cluster['cluster_id'], 'members': cluster['members'], 'scope': {'unit_type': 'group_window', 'boundary': '[start,end)', 'start': window['start'], 'end': window['end'], 'window_id': w}, 'records': sorted(records, key=lambda p: (timestamp(p['timestamp']), p['record_id']))}
            raw_state(packet)
            packets.append(packet)
    write_jsonl(out / 'packets.jsonl', packets)
    write_json(out / 'packet_exclusions.json', excluded)
    return packets


def load_annotations(path):
    result = {}
    for ann in read_jsonl(path):
        key = ann['post_key']
        if digest(ann['source']) != key:
            raise ValueError('Annotation source hash mismatch')
        if key in result and result[key] != ann:
            raise ValueError('Conflicting annotation for ' + key)
        result[key] = ann
    return result


def phase2(config, packets_path, out):
    packets = list(read_jsonl(packets_path))
    for case in packets:
        raw_state(case)
    if config.get('annotations'):
        annotations = load_annotations(config['annotations'])
        out.mkdir(parents=True, exist_ok=False)
        needed = {digest({k: r.get(k) for k in ('author', 'timestamp', 'text')}) for c in packets for r in c['records'] if r['kind'] == 'post'}
        write_jsonl(out / 'post_annotations.jsonl', [annotations[k] for k in sorted(needed & annotations.keys())])
        write_json(out / 'run_report.json', {'mode': 'existing_annotations', 'required_posts': len(needed), 'available_posts': len(needed & annotations.keys()), 'missing_posts': len(needed - annotations.keys()), 'network_requests': 0})
        return
    if not packets:
        out.mkdir(parents=True, exist_ok=False)
        write_jsonl(out / 'post_annotations.jsonl', [])
        write_json(out / 'run_report.json', {'mode': 'empty', 'network_requests': 0})
        return
    if not config.get('artifact_model'):
        raise ValueError('Set annotation.artifact_model to the actual Gemma model tag served by Ollama')
    args = annotation_parser().parse_args(['--input', str(packets_path), '--output-dir', str(out), '--sqlite', str(out.parent / 'annotations.sqlite'), '--artifact-model', config['artifact_model']])
    allowed = {'model', 'artifact_model', 'artifact_base_url', 'artifact_backend', 'artifact_context_tokens', 'artifact_source_chars', 'artifact_max_output_tokens', 'post_bytes', 'state_bytes', 'request_bytes', 'batch_posts', 'max_artifacts', 'timeout', 'min_probability', 'min_confidence', 'no_progress'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError('Unknown annotation options: ' + ', '.join(sorted(unknown)))
    for key, value in config.items():
        setattr(args, key, value)
    signature = digest(config)
    marker = out.parent / 'annotation_cache_config.json'
    if marker.exists() and read_json(marker)['sha256'] != signature:
        raise ValueError('Annotation settings changed; use a new run directory')
    write_json(marker, {'sha256': signature, 'configuration': config})
    annotate(args)


def categorize(result, policy):
    for group in result['target_groups']:
        propagation = group['influence_or_propagation_pattern'] == 'supported'
        shared = group['shared_target'] == 'supported' if policy == 'legacy_target' else group['shared_objective'] == 'supported'
        group['category'] = 'propagation' if propagation else 'shared_objective' if shared else 'insufficient_evidence'
        group['potentially_improper'] = any(len(authors) >= 2 for authors in group['improper_mechanism_agents'].values())
        group['supporting_agents'] = group['agents']
        group['support_fraction'] = len(group['agents']) / len(result['members'])
        group['mechanism_support'] = {k: {'agents': v, 'count': len(v), 'fraction': len(v) / len(result['members'])} for k, v in group['influence_mechanism_agents'].items()}
        group['objective_support'] = {k: {'agents': v, 'count': len(v), 'fraction': len(v) / len(result['members'])} for k, v in group['objective_author_groups'].items()}
    result['legacy_conclusion'] = result.pop('conclusion')
    result['category'] = max((g['category'] for g in result['target_groups']), key=PRIORITY.get, default='insufficient_evidence')
    result['potentially_improper'] = any(g['potentially_improper'] for g in result['target_groups'])
    result['category_policy'] = policy
    return result


def phase3(config, packets_path, annotations_path, clusters_path, out):
    from .io import probability
    threshold = probability(config.get('min_probability', .6))
    policy = config.get('objective_policy', 'legacy_target')
    if policy not in {'legacy_target', 'compatible_objective'}:
        raise ValueError('Unknown objective policy')
    annotations = load_annotations(annotations_path)
    cases = list(read_jsonl(packets_path))
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case ID')
    results = []
    for case in cases:
        raw_state(case)
        result = categorize(assess(case, annotations, threshold), policy)
        result['cluster_id'] = case['cluster_id']
        results.append(result)
    clusters = []
    for cluster in read_json(clusters_path):
        matching = [r for r in results if r['cluster_id'] == cluster['cluster_id']]
        clusters.append(dict(cluster, category=max((r['category'] for r in matching), key=PRIORITY.get, default='insufficient_evidence'), case_ids=[r['case_id'] for r in matching], potentially_improper=any(r['potentially_improper'] for r in matching), coverage_status='assessed' if matching else 'no_eligible_packet'))
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'case_assessments.jsonl', results)
    write_jsonl(out / 'cluster_assessments.jsonl', clusters)
    write_json(out / 'report.json', {'category_counts': dict(Counter(c['category'] for c in clusters)), 'cases': len(results), 'clusters': len(clusters), 'objective_policy': policy, 'minimum_probability_and_confidence': threshold, 'minimum_distinct_authors': 2, 'aggregation': 'highest category across cases; evidence is not pooled across windows', 'inputs_sha256': {str(p): sha256(p) for p in [packets_path, annotations_path, clusters_path]}})


def resolve_config(path):
    config = read_json(path)
    base = Path(path).resolve().parent
    for parent, key in [(config, 'windows'), (config, 'posts'), (config, 'output_dir'), *[(config.get('discovery', {}), k) for k in ['scores', 'embeddings', 'model']], (config.get('annotation', {}), 'annotations')]:
        if parent.get(key):
            parent[key] = str((base / parent[key]).resolve())
    return config


def run(config_path):
    config = resolve_config(config_path)
    out = Path(config['output_dir'])
    if out.exists():
        raise ValueError('Use a new output_dir; use individual phase commands to resume')
    windows = load_windows(config['windows'])
    inputs = [config['windows'], config['posts']]
    inputs += [config['discovery'][k] for k in ['scores', 'embeddings', 'model'] if config['discovery'].get(k)]
    if config.get('annotation', {}).get('annotations'):
        inputs.append(config['annotation']['annotations'])
    hashes = {p: sha256(p) for p in inputs}
    out.mkdir(parents=True)
    write_json(out / 'manifest.json', {'version': '0.1.0', 'configuration': config, 'input_sha256': hashes, 'status': 'running'})
    try:
        clusters = discover(config['discovery'], windows, out / 'phase1')
        make_packets(clusters, windows, config['posts'], out)
        phase2(config.get('annotation', {}), out / 'packets.jsonl', out / 'phase2')
        phase3(config.get('assessment', {}), out / 'packets.jsonl', out / 'phase2' / 'post_annotations.jsonl', out / 'phase1' / 'clusters.json', out / 'phase3')
    except Exception as exc:
        write_json(out / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
    write_json(out / 'manifest.json', {'version': '0.1.0', 'configuration': config, 'input_sha256': hashes, 'status': 'complete'})
    return out
