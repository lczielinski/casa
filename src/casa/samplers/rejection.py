from typing import List
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    InfNanRemoveLogitsProcessor,
)

from casa.samplers.base import BaseSampler, SamplingResult
from casa.utils.oracle_logits_processor import OracleLogitsProcessor
from casa.utils.grammar_logits_processor import GrammarLogitsProcessor
from casa.utils.scoring import get_seq_logprob_from_scores


class RS(BaseSampler):
    """Rejection sampling: generate, then retry on grammar violations."""

    def __init__(self, llm, grammar, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0):
        super().__init__(llm, grammar, max_new_tokens)
        self.learn_level = 0
        self.constrain_first = False
        self.constrain_all = False
        self.verbose = verbose
        self.temperature = temperature

    def _filter_generated_text(self, generated_ids):
        if generated_ids[0][-1] == self.llm.tokenizer.eos_token_id:
            return self.llm.tokenizer.decode(generated_ids[0][:-1])
        return self.llm.tokenizer.decode(generated_ids[0])

    def sample(
        self,
        prompt: str,
        n_samples: int = 1,
        max_attempts: int = 100,
    ) -> List[SamplingResult]:
        prompt_ids = self._encode_prompt(prompt)
        results = []
        # ARS/CARS/ASAP: each accepted program is masked out and never proposed
        # again. remove_generated() masks the exact token path; `seen` additionally
        # dedups on canonical text
        dedup = self.learn_level >= 2
        seen: set[str] = set()

        logits_processor = OracleLogitsProcessor(
            tokenizer=self.llm.tokenizer,
            grammar_constraint=self.grammar.recognizer,
            device=self.llm.device,
            learn_level=self.learn_level,
            constrain_first=self.constrain_first,
            constrain_all=self.constrain_all,
            temperature=self.temperature,
        )
        for sample_idx in range(n_samples):
            success = False

            for _ in range(1, max_attempts + 1):
                try:
                    result = self._generate_one(prompt_ids, logits_processor)
                except ValueError:
                    gen = logits_processor.generated_tokens
                    if dedup and (gen is None or len(gen) == 0):
                        if self.verbose:
                            print(f"[exhausted] {len(results)} distinct program(s)",
                                  flush=True)
                        return results
                    # Off-grammar: the processor still holds the rejected tokens.
                    if self.verbose:
                        rejected = self.llm.tokenizer.decode(
                            gen, skip_special_tokens=True
                        ).strip()
                        print(f"[reject] {rejected}", flush=True)
                    continue

                if dedup:
                    logits_processor.remove_generated()
                    key = result.text.strip()
                    if key in seen:
                        if self.verbose:
                            print(f"[dup] {key}", flush=True)
                        continue
                    seen.add(key)

                results.append(result)
                if self.verbose:
                    print(f"[{len(results)}/{n_samples}] {result.text.strip()}", flush=True)
                success = True
                break

            if not success:
                if dedup:
                    if self.verbose:
                        print(f"[exhausted] {len(results)} distinct program(s); "
                              f"none new in {max_attempts} attempts", flush=True)
                    break
                if self.verbose:
                    print(f"[timeout] sample {sample_idx + 1}: no valid program in "
                          f"{max_attempts} attempts", flush=True)

        return results

    def _generate_one(
        self,
        prompt_ids: torch.Tensor,
        logits_processor: OracleLogitsProcessor,
    ) -> SamplingResult:
        """Generate one sample; raises ValueError if it violates the grammar."""
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            eos_token_id=self.llm.tokenizer.eos_token_id,
            pad_token_id=self.llm.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=None,
        )

        logits_processor.reset()
        logits_processor_list = LogitsProcessorList([
            logits_processor,
            InfNanRemoveLogitsProcessor(),
        ])

        attention_mask = torch.ones_like(prompt_ids)

        output = self.llm.model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            tokenizer=self.llm.tokenizer,
            logits_processor=logits_processor_list,
        )

        output_ids = output.sequences
        raw_logprob = logits_processor.generation_ended(output_ids)

        generated_ids = output_ids[:, prompt_ids.shape[1]:]
        output_scores = torch.stack(output.scores, dim=1)

        constrained_logprob = get_seq_logprob_from_scores(
            output_scores,
            generated_ids,
            self.llm.tokenizer.eos_token_id,
        ).item()

        token_ids = generated_ids[0].tolist()
        tokens = [self.llm.tokenizer.decode([tid]) for tid in token_ids]
        text = self._filter_generated_text(generated_ids)

        return SamplingResult(
            tokens=tokens,
            token_ids=token_ids,
            text=text,
            raw_logprob=raw_logprob,
            constrained_logprob=constrained_logprob,
            success=True,
        )


class ARS(RS):
    """Adaptive rejection sampling: learns from rejected samples."""

    def __init__(self, llm, grammar, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0):
        super().__init__(llm, grammar, max_new_tokens, verbose, temperature)
        self.learn_level = 2


class RSFT(RS):
    """Rejection sampling with the first token constrained to the grammar."""

    def __init__(self, llm, grammar, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0):
        super().__init__(llm, grammar, max_new_tokens, verbose, temperature)
        self.learn_level = 0
        self.constrain_first = True


class CARS(RS):
    """Constrained adaptive rejection sampling: ARS plus a constrained first token."""

    def __init__(self, llm, grammar, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0):
        super().__init__(llm, grammar, max_new_tokens, verbose, temperature)
        self.learn_level = 3
        self.constrain_first = True


class ASAP(CARS):
    """Adaptive sampling with approximate futures (ASAP).

    CARS, but the grammar mask is applied at every step (not just the first
    token and revisited nodes), so only grammar-valid tokens are ever sampled
    and every generation is good. The trie still learns expected-future
    grammaticality, reweighting the constrained proposal toward the true
    grammar-aligned distribution.
    """

    def __init__(self, llm, grammar, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0):
        super().__init__(llm, grammar, max_new_tokens, verbose, temperature)
        self.constrain_all = True


class GCD(BaseSampler):
    """Grammar-constrained decoding (GCD).

    Mask the logits to grammar-valid tokens at every step and sample, with no
    trie, no learning, and no reweighting. Every output is grammar-valid, but the
    distribution is the locally normalized constrained model -- biased relative to
    the true grammar-aligned distribution that ARS/CARS/ASAP recover. This is the
    standard baseline. Samples are drawn independently (with replacement).
    """

    def __init__(self, llm, grammar, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0):
        super().__init__(llm, grammar, max_new_tokens)
        self.verbose = verbose
        self.temperature = temperature

    def _filter_generated_text(self, generated_ids):
        if generated_ids[0][-1] == self.llm.tokenizer.eos_token_id:
            return self.llm.tokenizer.decode(generated_ids[0][:-1])
        return self.llm.tokenizer.decode(generated_ids[0])

    def sample(
        self,
        prompt: str,
        n_samples: int = 1,
        max_attempts: int = 100,
    ) -> List[SamplingResult]:
        prompt_ids = self._encode_prompt(prompt)
        results = []

        for sample_idx in range(n_samples):
            success = False
            for _ in range(1, max_attempts + 1):
                try:
                    result = self._generate_one(prompt_ids)
                except ValueError:
                    # Only happens on truncation (hit max_new_tokens before the
                    # grammar reached an accepting state); retry.
                    if self.verbose:
                        print("[reject] incomplete program", flush=True)
                    continue

                results.append(result)
                if self.verbose:
                    print(f"[{len(results)}/{n_samples}] {result.text.strip()}", flush=True)
                success = True
                break

            if not success and self.verbose:
                print(f"[timeout] sample {sample_idx + 1}: no complete program in "
                      f"{max_attempts} attempts", flush=True)

        return results

    def _generate_one(self, prompt_ids: torch.Tensor) -> SamplingResult:
        """Generate one grammar-valid program; raises ValueError if truncated."""
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            eos_token_id=self.llm.tokenizer.eos_token_id,
            pad_token_id=self.llm.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=None,
        )

        self.grammar.reset()
        grammar_processor = GrammarLogitsProcessor(
            tokenizer=self.llm.tokenizer,
            grammar_constraint=self.grammar.recognizer,
            device=self.llm.device,
            prompt_length=len(prompt_ids[0]),
            temperature=self.temperature,
        )
        logits_processor_list = LogitsProcessorList([
            grammar_processor,
            InfNanRemoveLogitsProcessor(),
        ])

        output = self.llm.model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            generation_config=generation_config,
            tokenizer=self.llm.tokenizer,
            logits_processor=logits_processor_list,
        )

        generated_ids = output.sequences[:, prompt_ids.shape[1]:]
        # EOS is grammar-masked until an accepting state, so a complete program
        # leaves the matcher accepting; otherwise the program was truncated.
        if not self.grammar.recognizer.ll_matcher.is_accepting():
            raise ValueError("incomplete program")

        output_scores = torch.stack(output.scores, dim=1)
        logprob = get_seq_logprob_from_scores(
            output_scores,
            generated_ids,
            self.llm.tokenizer.eos_token_id,
        ).item()

        token_ids = generated_ids[0].tolist()
        tokens = [self.llm.tokenizer.decode([tid]) for tid in token_ids]
        text = self._filter_generated_text(generated_ids)

        # GCD has no separate unconstrained pass; report the constrained logprob.
        return SamplingResult(
            tokens=tokens,
            token_ids=token_ids,
            text=text,
            raw_logprob=logprob,
            constrained_logprob=logprob,
            success=True,
        )
