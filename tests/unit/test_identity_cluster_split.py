"""
Contract tests for TrainingDataSplitter (identity-cluster split).

Dependency type: NONE — in-memory LabeledPair lists.

The contract:
  1. every pair is in exactly one split
  2. all positive pairs of one match cluster land in the same split (no identity
     leakage of matches across train/val/test)
  3. each tenant is split independently and appears in every split
  4. label ratio per split stays close to the overall ratio
  5. deterministic for a seed; independent of pair orientation
  6. entity ids are tenant-scoped (the same id in two tenants is two entities)
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import pytest

from src.domain.models import LabeledPair
from src.pipeline.stages.data_collection import TrainingDataSplitter

T1, T2 = "tenant-1", "tenant-2"


def build(n_clusters=300, per_cluster=3, neg_ratio=5, tenants=(T1, T2), seed=7):
    """Clusters of `per_cluster` records -> all within-cluster positives, plus random
    cross-cluster negatives at neg_ratio:1. Returns (pairs, cluster_of_entity)."""
    rng = random.Random(seed)
    pairs, cluster_of = [], {}
    for t in tenants:
        clusters = []
        for c in range(n_clusters):
            ents = [f"{t}-c{c}-r{r}" for r in range(per_cluster)]
            clusters.append(ents)
            for e in ents:
                cluster_of[(t, e)] = (t, c)
            for i in range(per_cluster):
                for j in range(i + 1, per_cluster):
                    pairs.append(LabeledPair(ents[i], ents[j], t, 1, 1.0, "synthetic"))
        n_pos = sum(1 for p in pairs if p.tenant_id == t and p.label == 1)
        seen = set()
        while sum(1 for p in pairs if p.tenant_id == t and p.label == 0) < n_pos * neg_ratio:
            a, b = rng.sample(range(n_clusters), 2)
            e1, e2 = rng.choice(clusters[a]), rng.choice(clusters[b])
            k = tuple(sorted((e1, e2)))
            if k not in seen:
                seen.add(k)
                pairs.append(LabeledPair(e1, e2, t, 0, 1.0, "synthetic"))
    return pairs, cluster_of


@pytest.fixture(scope="module")
def data():
    pairs, cluster_of = build()
    splitter = TrainingDataSplitter(test_ratio=0.15, val_ratio=0.10)
    tr, va, te = splitter.split(pairs)
    return pairs, cluster_of, splitter, {"train": tr, "val": va, "test": te}


def split_of(splits):
    return {i: name for name, idx in splits.items() for i in idx}


def test_every_pair_in_exactly_one_split(data):
    pairs, _, _, splits = data
    all_idx = [i for idx in splits.values() for i in idx]
    assert sorted(all_idx) == list(range(len(pairs)))


def test_positive_pairs_of_one_cluster_share_a_split(data):
    """The identity-leakage guarantee."""
    pairs, cluster_of, _, splits = data
    where = split_of(splits)
    cluster_splits = defaultdict(set)
    for i, p in enumerate(pairs):
        if p.label == 1:
            cluster_splits[cluster_of[(p.tenant_id, p.entity_id_1)]].add(where[i])
    assert cluster_splits
    assert all(len(s) == 1 for s in cluster_splits.values())


def test_no_test_identity_has_a_positive_in_train(data):
    pairs, cluster_of, _, splits = data
    train_clusters = {
        cluster_of[(pairs[i].tenant_id, pairs[i].entity_id_1)]
        for i in splits["train"] if pairs[i].label == 1
    }
    for name in ("val", "test"):
        for i in splits[name]:
            if pairs[i].label == 1:
                assert cluster_of[(pairs[i].tenant_id, pairs[i].entity_id_1)] not in train_clusters


def test_split_sizes_near_target(data):
    pairs, _, _, splits = data
    n = len(pairs)
    assert len(splits["test"]) == pytest.approx(0.15 * n, rel=0.10)
    assert len(splits["val"]) == pytest.approx(0.10 * n, rel=0.10)


def test_label_ratio_preserved_per_split(data):
    pairs, _, _, splits = data
    overall = sum(p.label == 0 for p in pairs) / sum(p.label == 1 for p in pairs)
    for name, idx in splits.items():
        labels = Counter(pairs[i].label for i in idx)
        assert labels[1] > 0 and labels[0] > 0, f"{name} lacks a class"
        assert labels[0] / labels[1] == pytest.approx(overall, rel=0.25), name


def test_every_tenant_in_every_split_in_proportion(data):
    """The docstring used to claim tenant stratification without implementing it."""
    pairs, _, _, splits = data
    for name, idx in splits.items():
        tenants = Counter(pairs[i].tenant_id for i in idx)
        assert set(tenants) == {T1, T2}, name
        assert tenants[T1] / tenants[T2] == pytest.approx(1.0, rel=0.25), name


def test_deterministic_for_a_seed(data):
    pairs, _, _, splits = data
    again = TrainingDataSplitter(test_ratio=0.15, val_ratio=0.10).split(pairs)
    assert again == (splits["train"], splits["val"], splits["test"])


def test_different_seed_changes_assignment(data):
    pairs, _, _, splits = data
    other = TrainingDataSplitter(test_ratio=0.15, val_ratio=0.10, random_seed=99).split(pairs)
    assert other[2] != splits["test"]


def test_pair_orientation_does_not_change_the_split(data):
    pairs, _, _, splits = data
    flipped = [LabeledPair(p.entity_id_2, p.entity_id_1, p.tenant_id, p.label, 1.0, p.source)
               for p in pairs]
    assert TrainingDataSplitter(0.15, 0.10).split(flipped) == (
        splits["train"], splits["val"], splits["test"])


def test_same_entity_id_in_two_tenants_is_not_merged():
    """Entity ids are tenant-scoped: a shared id must not join two tenants' clusters."""
    pairs = []
    for t in (T1, T2):
        pairs += [LabeledPair("E1", "E2", t, 1, 1.0, "s"), LabeledPair("E1", "E3", t, 0, 1.0, "s")]
    splitter = TrainingDataSplitter(0.15, 0.10)
    splitter.split(pairs)
    # Two tenants x one cluster {E1,E2} each, plus E3's anchor... E1 < E3 so the
    # negatives anchor to E1's cluster: exactly two anchored clusters, one per tenant.
    assert splitter.last_stats["clusters"] == 2


def test_stats_report_residual_entity_overlap(data):
    """Negatives may share an entity record with train; this must be measured, not hidden."""
    pairs, _, splitter, splits = data
    s = splitter.last_stats
    assert s["train_pairs"] + s["val_pairs"] + s["test_pairs"] == len(pairs)
    assert s["test_positive"] + s["test_negative"] == s["test_pairs"]
    assert 0 <= s["test_pairs_sharing_train_entity"] <= s["test_pairs"]
    assert "val_pairs_sharing_train_entity" in s


def test_invalid_ratios_rejected():
    with pytest.raises(ValueError):
        TrainingDataSplitter(test_ratio=0.6, val_ratio=0.5)


def test_tiny_stratum_goes_to_train_not_test():
    """A single cluster must not end up as the only data in test."""
    pairs = [LabeledPair("A", "B", T1, 1, 1.0, "s")]
    tr, va, te = TrainingDataSplitter().split(pairs)
    assert tr == [0] and va == [] and te == []
