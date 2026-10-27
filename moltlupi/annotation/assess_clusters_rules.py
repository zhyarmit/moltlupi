from collections import Counter, defaultdict
import hashlib
import json
import math
from .post_level_rubric import INFLUENCE_QUESTIONS
VERSION = 'cluster-rules-v2-influence'
OBJECTIVE_ROLES = {'beneficiary', 'promoted_entity', 'action_destination'}
OBJECTIVES = {'engagement', 'adoption', 'technical', 'cultural', 'financial', 'project'}
IMPROPER = {'hide_affiliation', 'disguise_origin', 'false_identity_instruction', 'evade_oversight', 'override_authorization_request'}
PARTICIPATION = {'proof_requested', 'reciprocity_offer', 'replication_instruction', 'reward_condition'}
INFLUENCE = set(INFLUENCE_QUESTIONS) | PARTICIPATION
ACTIONS = {'request_follow', 'request_subscribe', 'request_vote', 'request_comment', 'request_repost', 'request_install', 'request_transfer', 'request_information', 'bulk_engagement'}

def objective_reason(entry, threshold):
    if not entry:
        return 'objective_missing'
    if entry.get('diagnostic'):
        return 'objective_diagnostic_error'
    if entry.get('accepted') is not True:
        return 'objective_unaccepted'
    selected = label(entry, threshold)
    if selected is None:
        return 'objective_failed_confidence_or_probability_validation'
    if selected not in OBJECTIVES:
        return 'objective_label_' + str(selected)
    return None

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()

def label(entry, threshold):
    if not entry or entry.get('accepted') is not True or entry.get('diagnostic'):
        return None
    answer = entry.get('answer', {})
    probs = answer.get('probabilities', {})
    choice = answer.get('choice')

    def probability(x):
        return isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) and (0 <= x <= 1)
    if not probs or not all((probability(v) for v in probs.values())):
        return None
    if abs(sum(probs.values()) - 1) > 0.010000000001:
        return None
    p, confidence = (probs.get(choice), answer.get('confidence'))
    if not probability(p) or not probability(confidence):
        return None
    if min(p, confidence) < threshold or p + 1e-09 < max(probs.values()):
        return None
    return choice

def name(text):
    return ' '.join(text.strip().lstrip('#').split()).casefold()

def assess(case, annotations, threshold=0.6, min_improper_agents=2, ignored=()):
    groups = defaultdict(list)
    exclusions, limitations, evaluated, seen = ([], [], [], set())
    members = set(case['members'])
    for record in case['records']:
        if record['kind'] != 'post':
            continue
        src = {k: record.get(k) for k in ('author', 'timestamp', 'text')}
        key = digest(src)
        if key in seen:
            continue
        seen.add(key)
        ref = {'post_key': key, 'record_id': record['record_id'], 'author': src['author']}

        def exclude(reason):
            exclusions.append(dict(ref, reason=reason))
        if src['author'] not in members:
            exclude('author_not_in_case_members')
            continue
        ann = annotations.get(key)
        if ann is None:
            exclude('missing_annotation')
            continue
        if ann.get('source') != src or ann.get('author') != src['author']:
            raise ValueError('Annotation/source mismatch for ' + key)
        if ann.get('artifact_status') != 'complete':
            exclude('artifact_extraction_incomplete')
            continue
        dims = ann.get('dimensions', {})
        labels = {k: label(v, threshold) for k, v in dims.items()}
        if labels.get('interpretive_context') != 'operative':
            exclude('nonoperative_or_uncertain_context')
            continue
        if labels.get('purpose') == 'criticism':
            exclude('criticism')
            continue
        objective = labels.get('stated_objective')
        objective_issue = objective_reason(dims.get('stated_objective'), threshold)
        arts = {}
        for artifact in ann.get('artifacts', []):
            literal, norm = (artifact.get('text', ''), artifact.get('normalized', ''))
            if name(literal) in {'moltbook', *(name(x) for x in ignored)}:
                continue
            if not norm or norm.startswith('invalid_url:') or (not literal) or (literal not in src['text']):
                continue
            if label(artifact, threshold) in OBJECTIVE_ROLES:
                arts[norm] = artifact
        if not arts:
            exclude('no_accepted_objective_artifact')
            continue
        flags = sorted((k for k in IMPROPER if labels.get(k) == 'present'))
        if labels.get('disclosure') == 'concealment_requested':
            flags.append('disclosure')
        participation = sorted((k for k in PARTICIPATION if labels.get(k) == 'present'))
        influence = sorted((k for k in INFLUENCE if labels.get(k) == 'present'))
        actions = sorted((k for k in ACTIONS if labels.get(k) == 'present'))
        if labels.get('purpose') == 'recruitment':
            influence.append('purpose=recruitment')
        influence = sorted(set(influence + actions))
        post_limits = []
        if objective_issue:
            post_limits.append(objective_issue)
        missing_new = sorted(set(INFLUENCE_QUESTIONS) - set(dims))
        if missing_new:
            post_limits.append('new_influence_dimensions_not_annotated')
        if ann.get('artifact_coverage', {}).get('source_truncated') or any((v.get('source_coverage', {}).get('source_truncated') for v in dims.values())):
            post_limits.append('truncated_source')
        if len(arts) > 1 and (flags or influence):
            post_limits.append('strategy_target_ambiguous_multiple_artifacts')
        if any((labels.get(k) is None or labels.get(k) == 'unclear' for k in IMPROPER | {'disclosure'})):
            post_limits.append('some_strategy_labels_uncertain_or_missing')
        for norm, artifact in arts.items():
            evidence_keys = set(flags + participation + actions + ['interpretive_context', 'stated_objective', 'purpose', 'behaviour_status']) | set(INFLUENCE_QUESTIONS)
            groups[norm].append(dict(ref, artifact=artifact, objective=objective if objective in OBJECTIVES else None, objective_issue=objective_issue, improper_dimensions=flags, improper_counted=bool(flags) and len(arts) == 1, influence_dimensions=influence, requested_actions=actions, missing_influence_dimensions=missing_new, participation_dimensions=participation, behaviour_status=labels.get('behaviour_status'), evidence_annotations={k: dims[k] for k in sorted(evidence_keys) if k in dims}, limitations=post_limits))
        evaluated.append(key)
        limitations.extend(post_limits)
    results = []
    for artifact, posts in sorted(groups.items()):
        agents = sorted({p['author'] for p in posts})
        improper_agents = sorted({p['author'] for p in posts if p['improper_counted']})
        mechanisms = {k: sorted({p['author'] for p in posts if k in p['influence_dimensions']}) for k in sorted({k for p in posts for k in p['influence_dimensions']})}
        improper_mechanisms = {k: sorted({p['author'] for p in posts if p['improper_counted'] and k in p['improper_dimensions']}) for k in sorted({k for p in posts for k in p['improper_dimensions']})}
        outcomes = {k: sorted({p['author'] for p in posts if p['objective'] == k}) for k in sorted({p['objective'] for p in posts if p['objective']})}
        shared = len(agents) >= 2
        qualifying = {k: v for k, v in improper_mechanisms.items() if len(v) >= min_improper_agents}
        candidate = shared and bool(qualifying)
        influence_pattern = shared and any((len(v) >= 2 for v in mechanisms.values()))
        limits = sorted({v for p in posts for v in p['limitations']})
        if len({annotations[p['post_key']]['source']['text'] for p in posts}) < len(posts):
            limits.append('duplicate_text_may_be_a_public_template')
        if len(outcomes) > 1:
            limits.append('multiple_stated_objectives_not_assumed_equivalent')
        reason = 'R4: shared target and same explicit improper mechanism meet distinct-author threshold; review participation' if candidate else 'R3: shared target and repeated influence mechanism; improper behavior not established' if influence_pattern else 'R2: shared target; influence or impropriety not established' if shared else 'R1: fewer than two eligible authors share this target'
        results.append(dict(artifact_key=artifact, objective_author_groups=outcomes, agents=agents, improper_strategy_agents=improper_agents, candidate_agents=sorted({a for authors in qualifying.values() for a in authors}), shared_target='supported' if shared else 'unresolved', shared_objective='supported' if any((len(v) >= 2 for v in outcomes.values())) else 'unresolved', influence_or_propagation_pattern='supported' if influence_pattern else 'not_established', potentially_improper_requested_behavior='supported_in_posts' if improper_agents else 'not_established', influence_mechanism_agents=mechanisms, improper_mechanism_agents=improper_mechanisms, participation_evidence=dict(requested_by=sorted({p['author'] for p in posts if p['behaviour_status'] == 'requested'}), claimed_completed_by=sorted({p['author'] for p in posts if p['behaviour_status'] == 'claimed_completed'}), observed_uptake='not_established', note='Behaviour status describes the post generally; no verified target-specific compliance or exposure is inferred.'), conclusion='collusion_candidate' if candidate else 'influence_or_propagation_pattern' if influence_pattern else 'shared_target_only' if shared else 'insufficient_evidence', rule=reason, supporting_posts=posts, limitations=limits, coordination='unresolved', strategy_to_target_link='post_cooccurrence_only'))
    verdict = 'collusion_candidate' if any((g['conclusion'] == 'collusion_candidate' for g in results)) else 'influence_or_propagation_pattern' if any((g['conclusion'] == 'influence_or_propagation_pattern' for g in results)) else 'shared_target_only' if any((g['conclusion'] == 'shared_target_only' for g in results)) else 'insufficient_evidence'
    candidates = [g for g in results if g['conclusion'] == 'collusion_candidate']
    return dict(case_id=case['case_id'], scope=case.get('scope'), members=case['members'], version=VERSION, conclusion=verdict, target_groups=results, candidate_subgroups=[dict(artifact_key=g['artifact_key'], agents=g['candidate_agents']) for g in candidates], coverage=dict(unique_posts=len(seen), evaluated_posts=len(evaluated), excluded_posts=len(exclusions)), excluded_posts=exclusions, limitations=sorted(set(limitations)), exclusion_counts=dict(Counter((p['reason'] for p in exclusions))), objective_uncertainty_counts=dict(Counter((issue for issue in limitations if issue.startswith('objective_')))), automatic_exclusion=False, interpretation='Annotation-based screening. A candidate is not confirmed collusion. Participation and strategy-target links require source review; other cluster members are not implicated automatically.')
