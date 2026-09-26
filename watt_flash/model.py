"""Encoder backbone read through the packed layout, plus an option scorer."""
import json
import os
from dataclasses import dataclass, field, fields
from typing import Dict

import torch
import torch.nn as nn

from .layout import build_masks


@dataclass
class WattConfig:
    name: str = "watt-flash"
    backbone: str = "jhu-clsp/mmBERT-small"
    max_len: int = 8192
    max_instruction_tokens: int = 256
    max_option_tokens: int = 64
    head_hidden: int = 0
    scorer_init: str = "mlm"            # "mlm": scorer shaped like a masked-LM prediction head
    dropout: float = 0.1
    temperature: Dict[str, float] = field(default_factory=dict)
    marker: str = "before"

    @classmethod
    def load(cls, path: str) -> "WattConfig":
        raw = json.load(open(os.path.join(path, "watt_config.json")))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


class EncoderBackbone(nn.Module):
    """ModernBERT given our own masks, so the sliding window follows position ids (see layout.build_masks)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.window = int(model.config.local_attention)
        self.hidden_size = int(model.config.hidden_size)
        model.config.reference_compile = False

    def forward(self, input_ids, position_ids, global_mask, local_mask):
        m = self.model
        if hasattr(m, "rotary_emb"):   # transformers 5
            masks = {"full_attention": global_mask, "sliding_attention": local_mask}
            return m(input_ids=input_ids, position_ids=position_ids, attention_mask=masks).last_hidden_state
        h = m.embeddings(input_ids=input_ids)   # transformers 4
        for layer in m.layers:
            h = layer(h, attention_mask=global_mask, sliding_window_mask=local_mask, position_ids=position_ids)[0]
        return m.final_norm(h)


class WattModel(nn.Module):
    def __init__(self, cfg: WattConfig, backbone: nn.Module):
        super().__init__()
        self.cfg, self.backbone = cfg, backbone
        d = backbone.hidden_size
        hd = cfg.head_hidden or d
        if cfg.scorer_init == "mlm":
            self.scorer = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        else:
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Dropout(cfg.dropout), nn.Linear(d, hd), nn.GELU(), nn.Linear(hd, 1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Logits [questions, max options]; padded option slots are -inf."""
        g, l = build_masks(batch["segment_ids"], batch["position_ids"], batch["real"], self.backbone.window)
        h = self.backbone(batch["input_ids"], batch["position_ids"], g, l)
        m = h.reshape(-1, h.shape[-1])[batch["marker_index"]]
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~batch["marker_mask"], float("-inf"))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "WattModel":
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel
        try:
            from transformers.initialization import no_init_weights
        except ImportError:   # transformers 4
            from transformers.modeling_utils import no_init_weights

        cfg = WattConfig.load(path)
        with no_init_weights():
            enc = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(path, "backbone")), attn_implementation="sdpa")
        model = cls(cfg, EncoderBackbone(enc))
        model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
        return model.to(device).eval()
