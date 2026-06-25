from dataclasses import dataclass
from typing import List, Optional, Literal, Union
import numpy as np
import torch
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    LogitsProcessor,
    InfNanRemoveLogitsProcessor,
)

from casa.samplers.base import BaseSampler, SamplingResult
from casa.utils.grammar_logits_processor import GrammarLogitsProcessor
from casa.utils.scoring import get_seq_logprob_from_scores


@dataclass
class MCMCStep:
    """One MCMC step: the current state, the proposal, and the decision."""
    current: SamplingResult
    proposal: SamplingResult
    acceptance_prob: float
    accepted: bool


class _RestrictorLogitsProcessor(LogitsProcessor):
    """Forces generation to follow answer_ids, recording each token's logprob.

    Used to score a fixed sequence under the unconstrained model.
    """

    def __init__(self, prompt_len: int, answer_ids: torch.LongTensor,
                 temperature: float = 1.0):
        self.prompt_len = prompt_len
        self.answer_ids = answer_ids
        self.temperature = temperature
        self.result = torch.empty(len(answer_ids))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pos = input_ids.size(1) - self.prompt_len
        assert 0 <= pos < self.answer_ids.size(0)

        if pos > 0:
            assert input_ids[0, -1] == self.answer_ids[pos - 1]

        tempered = scores / self.temperature if self.temperature != 1.0 else scores
        logprobs = torch.log_softmax(tempered.to(torch.get_default_dtype()), dim=-1)
        self.result[pos] = logprobs[0][self.answer_ids[pos]]

        # Force the target token to be the only option.
        scores = scores.clone()
        scores.fill_(float('-inf'))
        scores[0, self.answer_ids[pos]] = 0

        return scores


class MCMC(BaseSampler):
    """MCMC sampling with three proposal strategies.

    From "Constrained Sampling for Language Models Should Be Easy: An MCMC
    Perspective":
    - uniform: resample from a uniformly-random position
    - priority: resample from an entropy-weighted position
    - restart: always resample from the beginning
    """

    def __init__(
        self,
        llm,
        grammar,
        variant: Literal["uniform", "priority", "restart"] = "uniform",
        max_new_tokens: int = 512,
        verbose: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__(llm, grammar, max_new_tokens)

        if variant not in ["uniform", "priority", "restart"]:
            raise ValueError(
                f"Invalid variant '{variant}'. Must be 'uniform', 'priority', or 'restart'."
            )

        self.variant = variant
        self.verbose = verbose
        self.temperature = temperature

    def _filter_generated_text(self, generated_ids):
        if generated_ids[0][-1] == self.llm.tokenizer.eos_token_id:
            return self.llm.tokenizer.decode(generated_ids[0][:-1])
        return self.llm.tokenizer.decode(generated_ids[0])

    def _compute_sequence_logprob_constrained(
        self,
        scores: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> float:
        return get_seq_logprob_from_scores(
            scores,
            query_ids,
            self.llm.tokenizer.eos_token_id,
        ).item()

    def _compute_sequence_logprob_unconstrained(
        self,
        prompt_ids: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> float:
        """Exact logprob of query_ids under the unconstrained model."""
        generation_config = GenerationConfig(
            max_new_tokens=query_ids.shape[1],
            num_return_sequences=1,
            do_sample=False,
            eos_token_id=self.llm.tokenizer.eos_token_id,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )

        restrictor = _RestrictorLogitsProcessor(
            prompt_ids.size(1), query_ids[0], temperature=self.temperature
        )
        self.llm.model.generate(
            prompt_ids,
            generation_config=generation_config,
            tokenizer=self.llm.tokenizer,
            logits_processor=LogitsProcessorList([restrictor]),
        )

        return restrictor.result.sum().item()

    def sample(
        self,
        prompt: str,
        n_samples: int = 1,
        n_steps: int = 10,
        return_steps: bool = True,
    ) -> Union[List[SamplingResult], List[List[MCMCStep]]]:
        """Run n_samples independent chains for n_steps each.

        With return_steps=True, returns one list of MCMCStep per chain; with
        return_steps=False, returns each chain's final SamplingResult.
        """
        prompt_ids = self._encode_prompt(prompt)

        if return_steps:
            all_chains = []
        else:
            final_results = []

        for sample_idx in range(n_samples):
            current_ids, current_scores = self._generate_constrained(
                prompt_ids=prompt_ids,
                prefix_ids=None,
            )

            current_cons_logprob = self._compute_sequence_logprob_constrained(
                current_scores, current_ids
            )
            current_raw_logprob = self._compute_sequence_logprob_unconstrained(
                prompt_ids, current_ids
            )

            if self.verbose:
                print(f"[chain {sample_idx} init] "
                      f"{self._filter_generated_text(current_ids).strip()}", flush=True)

            chain_steps = [] if return_steps else None

            for step in range(n_steps):
                proposal_ids, proposal_scores, forward_logprob = self._propose_next_sequence(
                    prompt_ids=prompt_ids,
                    current_ids=current_ids,
                    current_scores=current_scores,
                )

                proposal_cons_logprob = self._compute_sequence_logprob_constrained(
                    proposal_scores, proposal_ids
                )
                proposal_raw_logprob = self._compute_sequence_logprob_unconstrained(
                    prompt_ids, proposal_ids
                )

                if torch.equal(current_ids, proposal_ids):
                    acceptance_prob = 1.0
                else:
                    reverse_logprob = self._compute_proposal_logprob(
                        current_ids=proposal_ids,
                        current_scores=proposal_scores,
                        next_ids=current_ids,
                        next_scores=current_scores,
                    )

                    log_accept_ratio = (
                        proposal_raw_logprob - current_raw_logprob +
                        reverse_logprob - forward_logprob
                    )
                    acceptance_prob = min(1.0, np.exp(log_accept_ratio))

                accepted = bool(np.random.rand() < acceptance_prob)

                if self.verbose:
                    tag = "accept" if accepted else "reject"
                    print(f"[chain {sample_idx} step {step}] {tag} "
                          f"p={acceptance_prob:.2f}: "
                          f"{self._filter_generated_text(proposal_ids).strip()}",
                          flush=True)

                if return_steps:
                    current_result = self._create_result_with_logprobs(
                        current_ids, prompt_ids, current_raw_logprob, current_cons_logprob
                    )
                    proposal_result = self._create_result_with_logprobs(
                        proposal_ids, prompt_ids, proposal_raw_logprob, proposal_cons_logprob
                    )

                    chain_steps.append(MCMCStep(
                        current=current_result,
                        proposal=proposal_result,
                        acceptance_prob=acceptance_prob,
                        accepted=accepted,
                    ))

                if accepted:
                    current_ids = proposal_ids
                    current_scores = proposal_scores
                    current_cons_logprob = proposal_cons_logprob
                    current_raw_logprob = proposal_raw_logprob

            if return_steps:
                all_chains.append(chain_steps)
            else:
                final_results.append(self._create_result_with_logprobs(
                    current_ids, prompt_ids, current_raw_logprob, current_cons_logprob
                ))

        return all_chains if return_steps else final_results

    def _create_result_with_logprobs(
        self,
        token_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        raw_logprob: float,
        cons_logprob: float,
    ) -> SamplingResult:
        token_list = token_ids[0].tolist()
        tokens = [self.llm.tokenizer.decode([tid]) for tid in token_list]
        text = self._filter_generated_text(token_ids)

        return SamplingResult(
            tokens=tokens,
            token_ids=token_list,
            text=text,
            raw_logprob=raw_logprob,
            constrained_logprob=cons_logprob,
            success=True,
        )

    def _generate_constrained(
        self,
        prompt_ids: torch.Tensor,
        prefix_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a grammar-constrained suffix after an optional prefix."""
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

        input_ids = prompt_ids
        if prefix_ids is not None:
            input_ids = torch.cat([prompt_ids, prefix_ids], dim=-1)

        output = self.llm.model.generate(
            input_ids,
            generation_config=generation_config,
            tokenizer=self.llm.tokenizer,
            logits_processor=logits_processor_list,
        )

        output_ids = output.sequences[:, input_ids.shape[1]:]
        output_scores = torch.stack(output.scores, dim=1)

        return output_ids, output_scores

    def _compute_resampling_distribution(
        self,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Distribution over which position to resample from, per variant."""
        seq_len = current_ids.shape[1]

        if self.variant == "restart":
            distr = torch.zeros(seq_len, dtype=torch.float32)
            distr[0] = 1.0

        elif self.variant == "uniform":
            distr = torch.ones(seq_len) / seq_len

        elif self.variant == "priority":
            logprobs = torch.log_softmax(current_scores, dim=-1)

            mask = torch.isfinite(logprobs)
            probs = torch.exp(logprobs)
            masked_contrib = torch.where(
                mask,
                probs * logprobs,
                torch.zeros_like(probs),
            )
            entropies = -torch.sum(masked_contrib, dim=-1)

            # Subtract 1 so zero-entropy positions get zero weight.
            distr = torch.exp(entropies[0]) - 1
            distr = distr / torch.sum(distr)

        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        distr = distr.unsqueeze(0)
        assert distr.shape == current_ids.shape
        assert torch.allclose(distr.sum(), torch.tensor(1.0))

        return distr

    def _propose_next_sequence(
        self,
        prompt_ids: torch.Tensor,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Resample a suffix from a sampled position; return (ids, scores, logprob)."""
        resample_distr = self._compute_resampling_distribution(
            current_ids, current_scores
        )
        resample_idx = np.random.choice(
            len(current_ids[0]),
            p=resample_distr[0].cpu().numpy(),
        )

        prefix_ids = current_ids[:, :resample_idx]
        prefix_scores = current_scores[:, :resample_idx]

        resample_ids, resample_scores = self._generate_constrained(
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
        )

        next_ids = torch.cat([prefix_ids, resample_ids], dim=-1)
        next_scores = torch.cat([prefix_scores, resample_scores], dim=1)

        proposal_logprob = self._compute_proposal_logprob(
            current_ids=current_ids,
            current_scores=current_scores,
            next_ids=next_ids,
            next_scores=next_scores,
        )

        return next_ids, next_scores, proposal_logprob

    def _compute_proposal_logprob(
        self,
        current_ids: torch.Tensor,
        current_scores: torch.Tensor,
        next_ids: torch.Tensor,
        next_scores: torch.Tensor,
    ) -> float:
        """Logprob of proposing next_ids from current_ids, summed over positions."""
        resample_distr = self._compute_resampling_distribution(
            current_ids, current_scores
        )

        # Longest common prefix: the proposal could only have branched after it.
        lcp_idx = 0
        for p, c in zip(next_ids[0], current_ids[0]):
            if p == c:
                lcp_idx += 1
            else:
                break

        max_resample_idx = min(lcp_idx + 1, len(current_ids[0]))

        proposal_logprob = -np.inf
        for i in range(max_resample_idx):
            idx_prob = resample_distr[0][i].item()
            if idx_prob == 0:
                continue

            idx_logprob = np.log(idx_prob)

            suffix_ids = next_ids[:, i:]
            suffix_scores = next_scores[:, i:]
            suffix_logprob = get_seq_logprob_from_scores(
                suffix_scores,
                suffix_ids,
                self.llm.tokenizer.eos_token_id,
            ).item()

            proposal_logprob = np.logaddexp(
                proposal_logprob,
                idx_logprob + suffix_logprob,
            )

        return proposal_logprob
