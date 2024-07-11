import random

from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import torch as t
from torch import Tensor
from transformer_lens.HookedTransformer import Output


class Methods:

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
        seq = Methods.generate_repeated_tokens(model, seq_len, batch)
        logits, cache = model.run_with_cache(seq)
        return seq, logits, cache
