"""Packed layout: the state is encoded once and every question shares the forward pass.

    [CLS] state ... [SEP] | <q1> [M] opt [M] opt ... [SEP] | <q2> [M] opt ... [SEP] | ...
    seg:  0 0 0 0 0 0 0      1 1 1 1 1 1 1 1 1 1 1 1 1       2 2 2 2 2 2 2 2 2
    pos:  0 1 2 ...   S-1    S S+1 S+2 ...                   S S+1 S+2 ...

Every token sees the state and its own segment. The state never sees a question, and every question starts at
position S, so an answer does not depend on the other questions of the request.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
NOUL_DEFAULTS = ("no, the statement does not hold", "yes, the statement holds")


class Overflow(ValueError):
    def __init__(self, needed: int, limit: int, state_tokens: int):
        self.needed, self.limit, self.state_tokens = needed, limit, state_tokens
        super().__init__("request needs %d tokens (%d of them state); the window is %d" % (needed, state_tokens, limit))


def serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def option_labels(q: Dict[str, Any]) -> List[str]:
    """Answer keys in the order of the returned probabilities."""
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def option_texts(q: Dict[str, Any]) -> List[str]:
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        return [k if v in (None, "") else "%s: %s" % (k, serialize(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, serialize(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["%s: %s" % (k, serialize(crit[k]) if crit.get(k) not in (None, "") else NOUL_DEFAULTS[i])
            for i, k in enumerate(("false", "true"))]


@dataclass
class Packed:
    input_ids: List[int]
    position_ids: List[int]
    segment_ids: List[int]          # 0 = state, k >= 1 = k-th question
    markers: List[List[int]]        # per question: sequence index of each option's marker
    qtypes: List[int]
    n_state: int

    def __len__(self) -> int:
        return len(self.input_ids)


class Layout:
    def __init__(self, tok, max_len: int = 8192, max_instruction_tokens: int = 256, max_option_tokens: int = 64,
                 marker: str = "before"):
        self.tok, self.max_len, self.marker = tok, max_len, marker
        self.max_instruction_tokens, self.max_option_tokens = max_instruction_tokens, max_option_tokens
        self.cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        self.sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
        self.marker_id = tok.mask_token_id if tok.mask_token_id is not None else self.sep
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else self.sep

    def _ids(self, text: str, limit=None) -> List[int]:
        if self.tok.mask_token:
            text = text.replace(self.tok.mask_token, " ")  # user text cannot forge a marker
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids if limit is None else ids[:limit]

    def encode_state(self, state: Any) -> List[int]:
        return [self.cls] + self._ids(serialize(state)) + [self.sep]

    def encode_question(self, q: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """(ids, offset of each option's marker within ids)"""
        ins = q.get("instructions")
        ins = "" if ins is None else serialize(ins)
        ids = self._ids("%s question: %s" % (q["type"], ins), self.max_instruction_tokens)
        offsets = []
        for text in option_texts(q):
            body = self._ids(" " + text, self.max_option_tokens)
            if self.marker == "before":
                offsets.append(len(ids))
                ids = ids + [self.marker_id] + body
            else:
                ids = ids + body + [self.marker_id]
                offsets.append(len(ids) - 1)
        return ids + [self.sep], offsets

    def pack(self, state_ids: Sequence[int], questions: Sequence[Dict[str, Any]]) -> Packed:
        ids, pos, seg = list(state_ids), list(range(len(state_ids))), [0] * len(state_ids)
        n_state, markers = len(state_ids), []
        for k, q in enumerate(questions, start=1):
            q_ids, offsets = self.encode_question(q)
            if n_state + len(q_ids) > self.max_len:
                raise Overflow(n_state + len(q_ids), self.max_len, n_state)
            markers.append([len(ids) + o for o in offsets])
            ids += q_ids
            pos += range(n_state, n_state + len(q_ids))
            seg += [k] * len(q_ids)
        return Packed(ids, pos, seg, markers, [QTYPES[q["type"]] for q in questions], n_state)

    def pack_request(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Packed:
        return self.pack(self.encode_state(state), list(questions.values()))


def build_masks(segment_ids: torch.Tensor, position_ids: torch.Tensor, real: torch.Tensor, window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Boolean attention masks [B, 1, L, L] for global and sliding-window layers; the window is measured in positions."""
    si, sj = segment_ids[:, :, None], segment_ids[:, None, :]
    allowed = ((sj == 0) | (sj == si)) & real[:, None, :]
    eye = torch.eye(segment_ids.shape[1], dtype=torch.bool, device=segment_ids.device)[None]
    allowed = allowed | (eye & ~real[:, :, None])  # padded rows attend to themselves
    near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= window // 2
    return allowed[:, None], (allowed & (near | eye))[:, None]


def collate(items: Sequence[Packed], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch. Row r of `marker_index` is one question, flattened across the batch."""
    B, L = len(items), max(len(p) for p in items)
    K = max(len(m) for p in items for m in p.markers)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    seg = torch.full((B, L), -1, dtype=torch.long)
    real = torch.zeros((B, L), dtype=torch.bool)
    rows = []
    for b, p in enumerate(items):
        n = len(p)
        ids[b, :n], pos[b, :n], seg[b, :n], real[b, :n] = (torch.tensor(p.input_ids), torch.tensor(p.position_ids),
                                                          torch.tensor(p.segment_ids), True)
        rows += [[b * L + i for i in m] + [-1] * (K - len(m)) for m in p.markers]
    marker_index = torch.tensor(rows, dtype=torch.long)
    return {"input_ids": ids, "position_ids": pos, "segment_ids": seg, "real": real,
            "marker_index": marker_index.clamp(min=0), "marker_mask": marker_index >= 0}
