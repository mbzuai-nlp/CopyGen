#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from model.matcher import SourceMatcher


@dataclass(frozen=True)
class CopyTrainingSignals:
    longest_copy_lengths: list[int]
    best_copy_starts: list[list[int]]
    match_lengths: list[int]
    delta_match_lengths: list[int]


def _build_sparse_lcp_rows(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> list[tuple[list[int], list[int]]]:
    positions_by_token: dict[int, list[int]] = {}
    for idx, tok in enumerate(source_ids):
        positions_by_token.setdefault(int(tok), []).append(idx)

    rows: list[tuple[list[int], list[int]]] = [([], []) for _ in range(len(target_ids))]
    next_row: dict[int, int] = {}
    for target_pos in range(len(target_ids) - 1, -1, -1):
        starts = positions_by_token.get(int(target_ids[target_pos]), [])
        if not starts:
            rows[target_pos] = ([], [])
            next_row = {}
            continue

        lengths: list[int] = []
        row_map: dict[int, int] = {}
        for start in starts:
            match_len = 1 + next_row.get(start + 1, 0)
            lengths.append(match_len)
            row_map[start] = match_len
        rows[target_pos] = (starts, lengths)
        next_row = row_map
    return rows


def build_longest_copy_targets(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> tuple[list[int], list[list[int]]]:
    rows = _build_sparse_lcp_rows(source_ids, target_ids)
    longest_lengths: list[int] = []
    best_starts: list[list[int]] = []

    for starts, lengths in rows:
        if not lengths:
            longest_lengths.append(0)
            best_starts.append([])
            continue
        best_len = max(lengths)
        longest_lengths.append(best_len)
        best_starts.append([start for start, length in zip(starts, lengths) if length == best_len])

    return longest_lengths, best_starts


def compute_prefix_match_features(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> tuple[list[int], list[int]]:
    matcher = SourceMatcher(source_ids)
    match_lengths: list[int] = []
    delta_lengths: list[int] = []

    for tok in target_ids:
        features = matcher.current_features()
        match_lengths.append(features.longest_match_length)
        delta_lengths.append(features.delta_match_length)
        matcher.append(int(tok))

    return match_lengths, delta_lengths


def build_training_signals(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> CopyTrainingSignals:
    longest_copy_lengths, best_copy_starts = build_longest_copy_targets(source_ids, target_ids)
    match_lengths, delta_match_lengths = compute_prefix_match_features(source_ids, target_ids)
    return CopyTrainingSignals(
        longest_copy_lengths=longest_copy_lengths,
        best_copy_starts=best_copy_starts,
        match_lengths=match_lengths,
        delta_match_lengths=delta_match_lengths,
    )
