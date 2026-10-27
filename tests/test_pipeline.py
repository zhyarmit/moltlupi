import ast
import copy
import io
import itertools
import json
import tokenize
from pathlib import Path

import numpy as np
import pytest

from moltlupi.annotation.assess_clusters_rules import assess, digest
from moltlupi.annotation.post_annotation_pipeline import parser as annotation_parser, run as annotate
from moltlupi.annotation.post_level_rubric import DIMENSIONS
from moltlupi.discovery import Scores, discover, sampled_pairs
from moltlupi.io import load_windows, read_json, read_jsonl, write_json, write_jsonl
from moltlupi.pipeline import categorize, make_packets, run


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'


def test_offline_end_to_end(tmp_path):
    config = read_json(EXAMPLES / 'config.json')
    config['windows'] = str(EXAMPLES / 'windows.json')
    config['posts'] = str(EXAMPLES / 'posts.jsonl')
    config['discovery']['scores'] = str(EXAMPLES / 'scores.jsonl')
    config['annotation']['annotations'] = str(EXAMPLES / 'annotations.jsonl')
    config['output_dir'] = str(tmp_path / 'run')
    path = tmp_path / 'config.json'
    write_json(path, config)
    out = run(path)
    assert read_json(out / 'phase3' / 'report.json')['category_counts'] == {'insufficient_evidence': 1, 'propagation': 1, 'shared_objective': 1}
    assert read_json(out / 'manifest.json')['status'] == 'complete'
    assert all(len(c['members']) == 4 for c in read_json(out / 'phase1' / 'clusters.json'))
    with pytest.raises(ValueError, match='new output_dir'):
        run(path)


def test_pair_sampling():
    agents = list('abcdef')
    all_pairs = set(itertools.combinations(agents, 2))
    sample = list(sampled_pairs(agents, 14, 12))
    assert len(set(sample)) == 14
    assert set(sample) <= all_pairs
    assert sample == list(sampled_pairs(agents, 14, 12))
    assert set(sampled_pairs(agents, None, 0)) == all_pairs


def test_onnx_pair_features_and_sigmoid(tmp_path):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    vectors = np.array([[1, 2], [3, 4]], dtype=np.float32)
    np.savez(tmp_path / 'z.npz', agent_id=np.array(['a', 'b']), window_id=np.array(['w0', 'w0']), joint_repr=vectors)
    weights = np.array([[.1], [.2], [.3], [.4]], dtype=np.float32)
    graph = helper.make_graph([helper.make_node('MatMul', ['x', 'weights'], ['y'])], 'test_only', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [None, 4])], [helper.make_tensor_value_info('y', TensorProto.FLOAT, [None, 1])], [numpy_helper.from_array(weights, 'weights')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.save(model, tmp_path / 'test.onnx')
    scores = Scores({'embeddings': str(tmp_path / 'z.npz'), 'model': str(tmp_path / 'test.onnx')}, load_windows(EXAMPLES / 'windows.json'))
    scores.complete([('b', 'a')])
    expected_logit = np.array([2, 2, 3, 8]) @ weights[:, 0]
    assert scores.aggregate(('a', 'b')) == pytest.approx(1 / (1 + np.exp(-expected_logit)))
    assert scores.calls == 1
    scores.complete([('a', 'b')])
    assert scores.calls == 1


def test_missing_and_conflicting_scores(tmp_path):
    windows = load_windows(EXAMPLES / 'windows.json')
    rows = list(read_jsonl(EXAMPLES / 'scores.jsonl'))[:6]
    rows.pop()
    write_jsonl(tmp_path / 'scores.jsonl', rows)
    scores = Scores({'scores': str(tmp_path / 'scores.jsonl')}, windows)
    with pytest.raises(ValueError, match='Missing score'):
        scores.complete(itertools.combinations(scores.agents, 2))
    rows.append(dict(rows[0]))
    write_jsonl(tmp_path / 'scores.jsonl', rows)
    with pytest.raises(ValueError, match='Duplicate'):
        Scores({'scores': str(tmp_path / 'scores.jsonl')}, windows)


def make_case():
    posts = [dict(p, kind='post') for p in list(read_jsonl(EXAMPLES / 'posts.jsonl'))[:4]]
    return {'case_id': 'test', 'members': [p['author'] for p in posts], 'records': posts, 'scope': {'unit_type': 'group_window', 'boundary': '[start,end)', 'start': '2026-01-01T00:00:00Z', 'end': '2026-01-02T00:00:00Z'}}


def test_live_annotation_contract_and_cache(tmp_path):
    case = make_case()
    write_jsonl(tmp_path / 'packets.jsonl', [case])
    args = annotation_parser().parse_args(['--input', str(tmp_path / 'packets.jsonl'), '--output-dir', str(tmp_path / 'out'), '--sqlite', str(tmp_path / 'cache.sqlite'), '--artifact-model', 'mock', '--no-progress'])
    calls = []
    def jev(payload):
        calls.append(payload)
        answers = {}
        for key, q in payload['questions'].items():
            criteria = q['criteria']
            choice = 'not_observed' if 'not_observed' in criteria else 'promoted_entity' if 'promoted_entity' in criteria else 'operative' if 'operative' in criteria else next(iter(criteria))
            answers[key] = {'type': 'choice', 'choice': choice, 'confidence': 1, 'probabilities': {k: float(k == choice) for k in criteria}}
        return {'model': 'mock', 'answers': answers}
    def gemma(payload):
        return {'done': True, 'done_reason': 'stop', 'message': {'content': '{"artifacts":[{"text":"DemoAlpha"}]}'}}
    annotate(args, transport=jev, artifact_transport=gemma)
    rows = list(read_jsonl(tmp_path / 'out' / 'post_annotations.jsonl'))
    assert len(rows) == 4
    assert all(len(r['dimensions']) == 28 and r['artifact_status'] == 'complete' for r in rows)
    assert all(r['artifacts'][0]['normalized'] == 'name:demoalpha' for r in rows)
    calls.clear()
    args.output_dir = str(tmp_path / 'again')
    annotate(args, transport=jev, artifact_transport=lambda p: pytest.fail('Unexpected extraction request'))
    assert not calls


def test_objective_policy_and_confidence():
    case = make_case()
    annotations = {a['post_key']: a for a in read_jsonl(EXAMPLES / 'annotations.jsonl')}
    for ann in annotations.values():
        for key in ['request_repost', 'stated_objective']:
            ann['dimensions'][key]['accepted'] = False
    result = assess(case, annotations)
    assert categorize(copy.deepcopy(result), 'legacy_target')['category'] == 'shared_objective'
    assert categorize(copy.deepcopy(result), 'compatible_objective')['category'] == 'insufficient_evidence'
    for ann in annotations.values():
        ann['dimensions']['interpretive_context']['answer']['confidence'] = .5
    assert categorize(assess(case, annotations), 'legacy_target')['category'] == 'insufficient_evidence'


def test_window_boundary_and_source_integrity(tmp_path):
    case = make_case()
    posts = case['records']
    posts.append(dict(posts[0], record_id='outside', timestamp='2026-01-02T00:00:00Z'))
    write_jsonl(tmp_path / 'posts.jsonl', posts)
    clusters = [{'cluster_id': 'c', 'members': case['members'], 'window_ids': ['w0']}]
    packets = make_packets(clusters, load_windows(EXAMPLES / 'windows.json'), tmp_path / 'posts.jsonl', tmp_path)
    assert len(packets[0]['records']) == 4
    annotations = {a['post_key']: a for a in read_jsonl(EXAMPLES / 'annotations.jsonl')}
    key = digest({k: case['records'][0][k] for k in ['author', 'timestamp', 'text']})
    annotations[key]['source']['text'] = 'tampered'
    with pytest.raises(ValueError, match='mismatch'):
        assess(case, annotations)


def test_no_code_comments_or_docstrings():
    for path in (ROOT / 'moltlupi').rglob('*.py'):
        code = path.read_text()
        assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(code).readline))
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node) is None
