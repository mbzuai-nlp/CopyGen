#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class MatchFeatures:
    longest_match_length: int
    delta_match_length: int


class SourceMatcher:
    """
    Online exact matcher for target-prefix suffixes against the source tokens.

    After consuming an output prefix y[:t], the matcher tracks all source end
    positions whose aligned source suffix equals a suffix of y[:t]. The longest
    such suffix length is the current grounding signal ell_t.

    When a source token merges trailing formatting (for example ".\\n\\n") but
    the model emits only a shorter visible prefix (for example "."), we also
    allow that emitted token to advance the frontier if its decoded text is a
    strict prefix of the next source token's decoded text. This makes the
    matcher robust to source-side merged boundary tokens without requiring
    full-sequence canonical repair.
    """

    def __init__(
        self,
        source_token_ids: Sequence[int],
        tokenizer: Any | None = None,
    ):
        self.source_token_ids = list(source_token_ids)
        self.source_length = len(self.source_token_ids)
        self.tokenizer = tokenizer
        self.positions_by_token: dict[int, list[int]] = {}
        for pos, tok in enumerate(self.source_token_ids):
            self.positions_by_token.setdefault(tok, []).append(pos)
        self.source_token_texts: list[str] | None = None
        self._token_text_cache: dict[int, str] = {}
        if self.tokenizer is not None:
            self.source_token_texts = [
                self.tokenizer.decode([tok], skip_special_tokens=False)
                for tok in self.source_token_ids
            ]
        self.reset()

    def reset(self) -> None:
        self._matched_ends: dict[int, int] = {}
        self._features = MatchFeatures(longest_match_length=0, delta_match_length=0)

    def current_features(self) -> MatchFeatures:
        return self._features

    def _decode_token_text(self, token_id: int) -> str:
        token_id = int(token_id)
        text = self._token_text_cache.get(token_id)
        if text is None:
            text = self.tokenizer.decode([token_id], skip_special_tokens=False) if self.tokenizer is not None else ""
            self._token_text_cache[token_id] = text
        return text

    def _advance_matches(self, token_id: int, matched_ends: dict[int, int] | None = None) -> tuple[dict[int, int], int]:
        if matched_ends is None:
            matched_ends = self._matched_ends
        next_matches: dict[int, int] = {}
        for pos in self.positions_by_token.get(int(token_id), []):
            next_matches[pos] = 1 + matched_ends.get(pos - 1, 0)

        if self.source_token_texts is not None and matched_ends:
            emitted_text = self._decode_token_text(token_id)
            if emitted_text:
                for prev_end, match_len in matched_ends.items():
                    next_pos = prev_end + 1
                    if next_pos >= self.source_length:
                        continue
                    source_text = self.source_token_texts[next_pos]
                    if not source_text or source_text == emitted_text:
                        continue
                    if source_text.startswith(emitted_text):
                        prev = next_matches.get(next_pos, 0)
                        if match_len + 1 > prev:
                            next_matches[next_pos] = match_len + 1
        longest = max(next_matches.values(), default=0)
        return next_matches, longest

    def append(self, token_id: int) -> MatchFeatures:
        prev_longest = self._features.longest_match_length
        token_id = int(token_id)
        next_matches, longest = self._advance_matches(token_id)
        self._matched_ends = next_matches
        self._features = MatchFeatures(
            longest_match_length=longest,
            delta_match_length=longest - prev_longest,
        )
        return self._features

    def extend(self, token_ids: Iterable[int]) -> MatchFeatures:
        features = self._features
        for token_id in token_ids:
            features = self.append(int(token_id))
        return features

    def candidate_next_starts(
        self,
        max_candidates: int = 8,
        longest_only: bool = True,
    ) -> list[int]:
        best_by_start: dict[int, int] = {}
        for end_pos, match_len in self._matched_ends.items():
            start = end_pos + 1
            if start >= self.source_length:
                continue
            prev = best_by_start.get(start, 0)
            if match_len > prev:
                best_by_start[start] = match_len

        if not best_by_start or max_candidates <= 0:
            return []

        ranked = sorted(best_by_start.items(), key=lambda item: (-item[1], item[0]))
        if longest_only:
            best_match_len = ranked[0][1]
            ranked = [item for item in ranked if item[1] == best_match_len]
        return [start for start, _ in ranked[:max_candidates]]

    def fallback_candidate_starts(
        self,
        preferred_token_ids: Sequence[int],
        max_candidates: int = 8,
    ) -> list[int]:
        if max_candidates <= 0:
            return []

        starts: list[int] = []
        seen: set[int] = set()
        for tok in preferred_token_ids:
            for pos in self.positions_by_token.get(int(tok), []):
                if pos in seen:
                    continue
                starts.append(pos)
                seen.add(pos)
                if len(starts) >= max_candidates:
                    return starts
        return starts
