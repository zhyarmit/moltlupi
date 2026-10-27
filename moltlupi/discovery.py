import itertools
import random
from collections import defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np

from .io import identifier, probability, read_jsonl, write_json, write_jsonl


class Scores:
    def __init__(self, config, windows):
        self.config = config
        self.windows = windows
        self.values = defaultdict(dict)
        self.agent_windows = defaultdict(set)
        self.embeddings = {}
        self.session = None
        self.calls = 0
        if config.get('embeddings'):
            with np.load(config['embeddings'], allow_pickle=False) as data:
                ids, win, vectors = data['agent_id'], data['window_id'], data['joint_repr']
                if vectors.ndim != 2 or len(ids) != len(win) or len(ids) != len(vectors) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid embedding arrays')
                if ids.dtype.kind not in 'US' or win.dtype.kind not in 'US':
                    raise ValueError('Embedding IDs must be string arrays')
                for a, w, z in zip(ids.astype(str), win.astype(str), vectors):
                    identifier(a)
                    if w not in windows or (a, w) in self.embeddings:
                        raise ValueError('Unknown or duplicate embedding window')
                    self.embeddings[a, w] = np.asarray(z, dtype=np.float32)
                    self.agent_windows[a].add(w)
        if config.get('scores'):
            for row in read_jsonl(config['scores']):
                a, b = identifier(row['agent_a']), identifier(row['agent_b'])
                w = identifier(row['window_id'])
                if a == b or w not in windows:
                    raise ValueError('Self-pair or unknown score window')
                pair = tuple(sorted((a, b)))
                if w in self.values[pair]:
                    raise ValueError('Duplicate pair/window score')
                self.values[pair][w] = probability(row['probability'])
                if self.embeddings and ((a, w) not in self.embeddings or (b, w) not in self.embeddings):
                    raise ValueError('Score has no matching agent/window embeddings')
                self.agent_windows[a].add(w)
                self.agent_windows[b].add(w)
        if config.get('model'):
            if not self.embeddings:
                raise ValueError('ONNX inference requires embeddings')
            import onnxruntime as ort
            self.session = ort.InferenceSession(config['model'], providers=['CPUExecutionProvider'])
            if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
                raise ValueError('Model must have one float32 feature input and one scalar-per-pair output')
        self.agents = sorted(self.agent_windows)
        if len(self.agents) < 2:
            raise ValueError('Need at least two eligible agents')

    def complete(self, pairs):
        batch = []
        for a, b in pairs:
            pair = tuple(sorted((a, b)))
            shared = self.agent_windows[a] & self.agent_windows[b]
            if not shared:
                continue
            for w in sorted(shared):
                if w not in self.values[pair]:
                    if self.session is None:
                        raise ValueError(f'Missing score for {pair} in {w}; provide a complete score file or model and embeddings')
                    batch.append((a, b, w))
                    if len(batch) >= self.config.get('batch_size', 512):
                        self.infer(batch)
                        batch = []
        if batch:
            self.infer(batch)

    def infer(self, batch):
        left = np.stack([self.embeddings[a, w] for a, b, w in batch])
        right = np.stack([self.embeddings[b, w] for a, b, w in batch])
        features = np.concatenate((np.abs(left - right), left * right), axis=1).astype(np.float32)
        result = np.asarray(self.session.run(None, {self.session.get_inputs()[0].name: features})[0])
        if result.shape not in {(len(batch),), (len(batch), 1)} or not np.isfinite(result).all():
            raise ValueError('Model returned an invalid shape or nonfinite scores')
        result = result.reshape(-1).astype(np.float64)
        if self.config.get('output_type', 'logits') == 'logits':
            result = np.exp(-np.logaddexp(0, -result))
        for (a, b, w), value in zip(batch, result):
            self.values[tuple(sorted((a, b)))][w] = probability(float(value))
        self.calls += 1

    def aggregate(self, pair):
        vals = self.values.get(tuple(sorted(pair)), {})
        shared = self.agent_windows[pair[0]] & self.agent_windows[pair[1]]
        if not shared or set(vals) != shared:
            return None
        scores = [vals[w] for w in sorted(shared)]
        return max(scores) if self.config.get('aggregation', 'mean') == 'max' else sum(scores) / len(scores)


def sampled_pairs(agents, count, seed):
    n = len(agents)
    total = n * (n - 1) // 2
    if count is None or count >= total:
        return itertools.combinations(agents, 2)
    ranks = random.Random(seed).sample(range(total), count)
    def decode(rank):
        lo, hi = 0, n - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if mid * (2 * n - mid - 1) // 2 <= rank:
                lo = mid
            else:
                hi = mid
        offset = lo * (2 * n - lo - 1) // 2
        return agents[lo], agents[lo + 1 + rank - offset]
    return (decode(rank) for rank in sorted(ranks))


def discover(config, windows, out):
    threshold = probability(config.get('edge_threshold', .85))
    density_threshold = probability(config.get('density_threshold', threshold))
    min_density = probability(config.get('min_density', .80))
    min_size = config.get('min_size', 4)
    if not isinstance(min_size, int) or min_size < 2:
        raise ValueError('min_size must be at least two')
    if config.get('aggregation', 'mean') not in {'mean', 'max'}:
        raise ValueError('aggregation must be mean or max')
    if config.get('output_type', 'logits') not in {'logits', 'probabilities'}:
        raise ValueError('output_type must be logits or probabilities')
    if config.get('batch_size', 512) < 1 or config.get('resolution', .1) <= 0:
        raise ValueError('Batch size and resolution must be positive')
    count = config.get('sample_pairs', 10000000)
    if count is not None and (not isinstance(count, int) or count < 1):
        raise ValueError('sample_pairs must be positive or null for all pairs')
    scores = Scores(config, windows)
    if scores.session:
        initial = list(sampled_pairs(scores.agents, count, config.get('seed', 0)))
        scores.complete(initial)
    else:
        initial = sorted(scores.values)
    index = {a: i for i, a in enumerate(scores.agents)}
    edges, weights = [], []
    for pair in initial:
        value = scores.aggregate(pair)
        if value is not None and value >= threshold and value > 0:
            edges.append((index[pair[0]], index[pair[1]]))
            weights.append(value)
    graph = ig.Graph(n=len(index), edges=edges, directed=False)
    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights=weights, resolution_parameter=config.get('resolution', .1), seed=config.get('seed', 0), n_iterations=10)
    groups = sorted(sorted(scores.agents[i] for i in part) for part in partition if len(part) >= min_size)
    clusters, rejected = [], []
    for members in groups:
        pairs = list(itertools.combinations(members, 2))
        scores.complete(pairs)
        values = [scores.aggregate(pair) for pair in pairs]
        if any(v is None for v in values):
            rejected.append({'members': members, 'reason': 'incomplete_pair_evidence'})
            continue
        density = sum(v >= density_threshold for v in values) / len(pairs)
        if density < min_density:
            rejected.append({'members': members, 'reason': 'density', 'density': density})
            continue
        common = set.intersection(*(scores.agent_windows[a] for a in members))
        cluster_id = f'cluster_{len(clusters):05d}'
        clusters.append({'cluster_id': cluster_id, 'members': members, 'size': len(members), 'density': density, 'mean_score': sum(values) / len(values), 'window_ids': sorted(common)})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'clusters.json', clusters)
    write_json(out / 'rejected.json', rejected)
    write_jsonl(out / 'pair_scores.jsonl', ({'agent_a': a, 'agent_b': b, 'window_id': w, 'probability': v} for (a, b), vals in sorted(scores.values.items()) for w, v in sorted(vals.items())))
    write_json(out / 'report.json', {'agents': len(index), 'initial_pairs': len(initial), 'graph_edges': len(edges), 'retained_clusters': len(clusters), 'onnx_batches': scores.calls, 'configuration': config})
    return clusters
