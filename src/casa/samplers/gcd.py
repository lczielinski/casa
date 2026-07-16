from typing import List
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    InfNanRemoveLogitsProcessor,
)

from casa.samplers.base import BaseSampler, SamplingResult
from casa.utils.grammar_logits_processor import GrammarLogitsProcessor
from casa.utils.scoring import get_seq_logprob_from_scores
from casa.utils.helpers import print_progress


class GCD(BaseSampler):
    """Grammar-Constrained Decoding (GCD).

    Masks grammar-invalid tokens at each decoding step. Rejection-free and fast,
    but distorts the LM distribution (see ASAp for the de-biased variant).
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
        """Generate samples using grammar-constrained decoding.

        Args:
            prompt: Input prompt.
            n_samples: Number of samples to generate.
            max_attempts: Maximum attempts per sample.
        """
        prompt_ids = self._encode_prompt(prompt)
        results = []

        for sample_idx in range(n_samples):
            n_attempts = 0
            success = False

            for attempt in range(max_attempts):
                n_attempts += 1

                try:
                    result = self._generate_one(prompt_ids)
                except ValueError:
                    continue
                if result is None:
                    continue

                result.attempts = n_attempts
                results.append(result)
                print_progress(sample_idx + 1, n_samples, n_attempts, max_attempts, self.verbose, timeout=False)
                success = True
                break

            if not success:
                print_progress(sample_idx + 1, n_samples, n_attempts, max_attempts, self.verbose, timeout=True)

        return results

    def _generate_one(self, prompt_ids: torch.Tensor) -> SamplingResult:
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=self.temperature,
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
            prompt_length=prompt_ids.shape[1],
        )
        logits_processor_list = LogitsProcessorList([
            grammar_processor,
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

        generated_ids = output.sequences[:, prompt_ids.shape[1]:]

        recognizer = self.grammar.recognizer
        if not recognizer.try_advance_token_ids(generated_ids[0]):
            return None
        if generated_ids[0][-1] != self.llm.tokenizer.eos_token_id:
            if not recognizer.is_accepting():
                return None

        output_scores = torch.stack(output.scores, dim=1)
        constrained_logprob = get_seq_logprob_from_scores(
            output_scores,
            generated_ids,
            self.llm.tokenizer.eos_token_id,
        ).item()
        raw_logprob = self._compute_raw_logprob(prompt_ids, generated_ids)

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

    def _compute_raw_logprob(self, prompt_ids: torch.Tensor, generated_ids: torch.Tensor) -> float:
        full = torch.cat([prompt_ids, generated_ids], dim=1)
        with torch.no_grad():
            logits = self.llm.model(full).logits[0].float()
        if self.temperature != 1.0:
            logits = logits / self.temperature
        logprobs = torch.log_softmax(logits, dim=-1)
        start = prompt_ids.shape[1]
        seq = generated_ids[0]
        return sum(logprobs[start - 1 + j, seq[j]].item() for j in range(len(seq)))
