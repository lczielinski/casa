"""CARS: Constrained Adaptive Rejection Sampling.

The default implementation decodes token-by-token with:

  * a within-attempt KV cache (incremental decoding), and
  * a cross-attempt oracle trie that caches each prefix's next-token
    distribution, so revisited prefixes skip the model forward entirely.

This is faster than calling ``model.generate`` once per attempt (which re-runs a
full forward over the whole prompt every attempt and cannot reuse work across
attempts) and is verified behaviourally equivalent to it: same target
distribution, same ``raw_logprob`` accounting (sum of model log-probs over all
generated tokens, including EOS), and the same constrained-distribution
log-probs. Pass ``fast=False`` to use the original ``generate``-based path.

The grammar backend (xgrammar / llguidance) is selected on the :class:`Grammar`
object; this sampler is engine-agnostic and goes through the recognizer's common
interface (``filter_vocab`` / ``apply_token_bitmask`` / ``is_accepting`` / ...).
"""

import time
from typing import List, Optional

import torch

from casa.samplers.base import SamplingResult
from casa.utils.oracle_trie import Trie
from casa.utils.helpers import print_progress


class CARS:
    """Constrained Adaptive Rejection Sampling (HF backend, KV-cached by default)."""

    def __init__(self, llm, grammar, max_new_tokens: int = 512,
                 verbose: bool = False, temperature: float = 1.0, fast: bool = True):
        self.llm = llm
        self.grammar = grammar
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.temperature = temperature
        self.fast = fast
        self.device = llm.device
        self.eos_token_id = llm.tokenizer.eos_token_id

        # Real vocabulary size; ids >= this are padding/phantom rows in the lm_head
        # (e.g. Qwen pads to a multiple of 128) and must never be sampled.
        self.vocab_size = grammar.recognizer.vocab_size

        self.trie = Trie()
        self.logits_process_time = 0.0

    # ------------------------------------------------------------------ #
    def _encode_prompt(self, prompt: str) -> List[int]:
        formatted = self.llm.format_prompt(prompt)
        return self.llm.tokenizer.encode(formatted, add_special_tokens=False)

    def _forward(self, token_ids: List[int], past, cached_len: int):
        """Return next-token logits for ``token_ids`` using an incremental KV cache.

        Only the tokens beyond ``cached_len`` are fed through the model; ``past``
        holds the KV cache for the first ``cached_len`` tokens.
        """
        if past is not None and cached_len > 0:
            new_ids = torch.tensor([token_ids[cached_len:]], device=self.device)
        else:
            new_ids = torch.tensor([token_ids], device=self.device)
            past = None
        with torch.no_grad():
            out = self.llm.model(new_ids, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1, :].float()
        return logits, out.past_key_values, len(token_ids)

    # ------------------------------------------------------------------ #
    def sample(self, prompt: str, n_samples: int = 1,
               max_attempts: int = 100) -> List[SamplingResult]:
        if not self.fast:
            # Original generate()-based reference implementation.
            from casa.samplers.rejection import _GenerateCARS
            return _GenerateCARS(
                self.llm, self.grammar, self.max_new_tokens,
                verbose=self.verbose, temperature=self.temperature,
            ).sample(prompt, n_samples, max_attempts)

        prompt_ids = self._encode_prompt(prompt)
        results: List[SamplingResult] = []
        # Fresh adaptive memory per sample() call (matches the generate()-based path,
        # which builds a fresh OracleLogitsProcessor/trie per call).
        self.trie = Trie()

        for sample_idx in range(n_samples):
            n_attempts = 0
            success = False
            for _ in range(max_attempts):
                n_attempts += 1
                result = self._generate_one(prompt_ids)
                if result is not None:
                    result.attempts = n_attempts
                    results.append(result)
                    print_progress(sample_idx + 1, n_samples, n_attempts,
                                   max_attempts, self.verbose, timeout=False)
                    success = True
                    break
            if not success:
                print_progress(sample_idx + 1, n_samples, n_attempts,
                               max_attempts, self.verbose, timeout=True)

        return results

    # ------------------------------------------------------------------ #
    def _generate_one(self, prompt_ids: List[int]) -> Optional[SamplingResult]:
        start_time = time.time()
        rec = self.grammar.recognizer
        rec.reset()

        past, cached_len = None, 0
        context: List[int] = []
        node = self.trie.root
        depth = 0
        raw_lps: List[float] = []     # log P_model(token) summed -> raw_logprob
        cons_lps: List[float] = []    # log P(sampling dist)(token) -> constrained_logprob
        recompute_needed = False

        for step in range(self.max_new_tokens):
            is_root = step == 0

            # 1) advance the grammar parser with the most recent token
            if not is_root:
                if not rec.try_advance_token_ids(torch.tensor(context)):
                    self._reject(node, depth, context, context[-1])
                    self.logits_process_time += time.time() - start_time
                    return None

                # 2) descend into the trie (creating the node if new)
                last = context[-1]
                if last not in node.children:
                    node.create_child(last)
                node = node.children[last]
                depth += 1

            # 3) obtain this node's next-token model log-prob distribution
            if node.raw_logprob is not None:
                raw = node.raw_logprob[0].to(self.device)
                is_new_node = False
            else:
                logits, past, cached_len = self._forward(
                    prompt_ids + context, past, cached_len)
                # Bug fix: mask padding/phantom ids beyond the real vocab.
                if logits.shape[-1] > self.vocab_size:
                    logits[self.vocab_size:] = float("-inf")
                # Bug fix: apply sampling temperature to the model logits only.
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                raw = torch.log_softmax(logits, dim=-1)
                node.raw_logprob = raw.unsqueeze(0).cpu()
                node.log_theta = torch.zeros(1, raw.shape[-1])
                # store grammar mask for this node (applied on revisits / at root)
                rec.apply_token_bitmask(node.log_theta, rec.filter_vocab())
                is_new_node = True
                recompute_needed = True

            # 4) build the sampling distribution (matches OracleLogitsProcessor:
            #    constrain at the root and on every *revisited* node; sample the
            #    raw model on a freshly-seen non-root node)
            adjust = is_root or not is_new_node
            if adjust:
                sampling_logits = raw + node.log_theta[0].to(self.device)
            else:
                sampling_logits = raw

            probs = torch.softmax(sampling_logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() < 1e-10:
                # node exhausted (no valid continuation) -> reject this path
                self.logits_process_time += time.time() - start_time
                return None
            next_token = torch.multinomial(probs, 1).item()

            # 5) accounting (raw = model; constrained = the dist we sampled from)
            raw_lps.append(raw[next_token].item())
            cons_lps.append(torch.log_softmax(sampling_logits, dim=-1)[next_token].item())

            # 6) EOS handling -> accept iff the grammar can consume the EOS here.
            #    (try_advance_token_ids only consumes EOS in an accepting state, so
            #    this matches the generate()-based path, which accepts an
            #    EOS-terminated sequence whenever the parser consumes the final EOS.)
            if next_token == self.eos_token_id:
                if rec.try_advance_token_ids(torch.tensor(context + [next_token])):
                    if recompute_needed:
                        self._recompute(node, depth, context)
                    self.logits_process_time += time.time() - start_time
                    return self._make_result(context, raw_lps, cons_lps)
                self._reject(node, depth, context, next_token)
                self.logits_process_time += time.time() - start_time
                return None

            context.append(next_token)

        # Ran out of budget without sampling EOS. Mirror generation_ended:
        # advance the parser with the final token, then accept a (non-EOS) sequence
        # iff the grammar is in an accepting state; otherwise mark the final token
        # failed (-inf) and reject.
        if not rec.try_advance_token_ids(torch.tensor(context)):
            self._reject(node, depth, context, context[-1])
            self.logits_process_time += time.time() - start_time
            return None
        if rec.is_accepting():
            if recompute_needed:
                self._recompute(node, depth, context)
            self.logits_process_time += time.time() - start_time
            return self._make_result(context, raw_lps, cons_lps, eos=False)
        self._reject(node, depth, context, context[-1])
        self.logits_process_time += time.time() - start_time
        return None

    # ------------------------------------------------------------------ #
    def _make_result(self, context, raw_lps, cons_lps, eos: bool = True) -> SamplingResult:
        # EOS-terminated samples include the EOS id (and its log-prob is already in
        # raw_lps/cons_lps); budget-truncated samples do not.
        token_ids = context + [self.eos_token_id] if eos else list(context)
        tokens = [self.llm.tokenizer.decode([t]) for t in token_ids]
        text = self.llm.tokenizer.decode(context)
        return SamplingResult(
            tokens=tokens,
            token_ids=token_ids,
            text=text,
            raw_logprob=float(sum(raw_lps)),
            constrained_logprob=float(sum(cons_lps)),
            success=True,
        )

    def _reject(self, node, depth, context, failed_token) -> None:
        node.log_theta[0, failed_token] = float("-inf")
        self._recompute(node, depth, context)

    def _recompute(self, node, depth, context) -> None:
        """Propagate constrained log-mass up the trie (the adaptive update)."""
        while depth > 0:
            new_log_theta = torch.log(
                torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum()
            )
            depth -= 1
            node = node.parent
            node.log_theta[0, context[depth]] = new_log_theta
