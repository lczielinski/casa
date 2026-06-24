import torch
from typing import List, Tuple


def get_seq_logprob_from_scores(
    scores: torch.Tensor,
    query_ids: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Sum token logprobs per sequence, up to and including the first EOS.

    scores: (batch, seq_len, vocab); query_ids: (batch, seq_len).
    Returns logprobs of shape (batch,).
    """
    assert scores.shape[0] == query_ids.shape[0], "Batch sizes must match"
    assert scores.shape[1] == query_ids.shape[1], "Sequence lengths must match"

    logprobs = torch.log_softmax(scores, dim=-1)

    batch_size, seq_len = query_ids.shape
    result = torch.zeros(batch_size, device=scores.device)

    for i in range(batch_size):
        seq_logprobs = logprobs[i, torch.arange(seq_len), query_ids[i]]

        eos_positions = torch.nonzero(query_ids[i] == eos_token_id)
        if eos_positions.shape[0] > 0:
            first_eos_pos = eos_positions[0].item()
            result[i] = seq_logprobs[:first_eos_pos + 1].sum()
        else:
            result[i] = seq_logprobs.sum()

    return result


def unbatch_sequences(
    sequences: torch.Tensor,
    eos_token_id: int,
) -> List[torch.Tensor]:
    """Split a batch into sequences, truncating each at its first EOS."""
    result = []
    for i in range(sequences.shape[0]):
        eos_idx = torch.nonzero(sequences[i] == eos_token_id)
        if eos_idx.shape[0] > 0:
            result.append(sequences[i][:eos_idx[0].item() + 1])
        else:
            result.append(sequences[i])
    return result


def scores_to_top_k(
    scores: torch.Tensor,
    k: int,
) -> List[List[Tuple[int, float]]]:
    """Top-k (token_id, logprob) pairs at each position of scores."""
    result = []
    for step_scores in scores:
        log_probs = torch.log_softmax(step_scores, dim=-1)
        top_probs, top_token_ids = torch.topk(log_probs, k=k)
        result.append(list(zip(top_token_ids.tolist(), top_probs.tolist())))
    return result
