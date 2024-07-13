import random

import einops
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import torch as t
from torch import Tensor
from transformer_lens.HookedTransformer import Output
from transformer_lens.hook_points import HookPoint


class InductionTask:
    @staticmethod
    def generate_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Int[
        Tensor, "batch full_seq_len"]:
        seq = t.empty(batch, seq_len, dtype=t.long)
        for i in range(batch):
            for j in range(seq_len):
                seq[i, j] = random.randint(0, model.cfg.d_vocab)
        bos = t.full((batch, 1), 50256, dtype=t.long)
        rep_seq = t.cat([bos, seq, seq], dim=1)
        return rep_seq

    @staticmethod
    def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> tuple[
        Tensor, Output, ActivationCache]:
        seq = InductionTask.generate_repeated_tokens(model, seq_len, batch)
        logits, cache = model.run_with_cache(seq)
        return seq, logits, cache


class InductionPatternDetector:
    @staticmethod
    def get_heads_with_pattern(cache: ActivationCache, model: HookedTransformer, threshold: float = 0.3) -> list:
        def get_pattern_score(head):
            seq_len = (head.shape[-1] - 1) // 2
            return head.diagonal(-seq_len + 1).mean()

        def get_pattern_for_head(head):
            pattern_score = get_pattern_score(head)
            return pattern_score > threshold

        heads_matching_pattern = []
        for layer in range(model.cfg.n_layers):
            heads = t.unbind(cache["pattern", layer])
            for head in range(len(heads)):
                if get_pattern_for_head(heads[head]):
                    heads_matching_pattern.append(f"{layer}.{head}")
        return heads_matching_pattern


class InductionHook:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
        self.filter = lambda name: name.endswith("pattern")

    def hook(self, pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint):
        hook.layer()
        batch, heads, dest, source = pattern.shape
        seq_len = dest // 2
        for head_idx in range(heads):
            head = pattern[:, head_idx, :, :]
            score = self.induction_pattern(head, seq_len)
            self.score_store[hook.layer(), head_idx] = score

    def induction_pattern(self, head: Float[Tensor, "batch dest_pos source_pos"], seq_len: int):
        return head.diagonal(1 - seq_len, dim1=-2, dim2=-1).mean()


class InductionAttribution:
    @staticmethod
    def logit_attribution(embed: Float[Tensor, "seq d_model"], results: List[Float[Tensor, "seq nheads d_model"]],
        W_U: Float[Tensor, "d_model d_vocab"], tokens: Int[Tensor, "seq"]) -> Float[Tensor, "seq-1 n_components"]:
        W_U_correct_tokens = W_U[:, tokens[1:]]
        direct_path = einops.einsum(embed[:-1], W_U_correct_tokens, "seq d_model, d_model seq -> seq")
        layers = [einops.einsum(result[:-1], W_U_correct_tokens, "seq nheads d_model, d_model seq -> seq nheads") for result in results]
        return t.cat([direct_path.unsqueeze(-1)] + layers, dim=-1)

    @staticmethod
    def get_results(cache: ActivationCache, model: HookedTransformer) -> List[Float[Tensor, "seq nheads d_model"]]:
        return [einops.einsum(cache["z", i], model.blocks[i].attn.W_O,
                              "seq head_idx d_head, head_idx d_head d_model -> seq head_idx d_model") for i in
                range(model.cfg.n_layers)]
