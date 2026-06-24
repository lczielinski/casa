import torch
from typing import Optional, Dict


class TrieNode:
    """A trie node caching a prefix's raw logprobs and learned log_theta mask."""

    def __init__(self, parent: Optional['TrieNode'] = None):
        self.parent = parent
        self.children: Dict[int, 'TrieNode'] = {}
        self.raw_logprob: Optional[torch.Tensor] = None
        self.log_theta: Optional[torch.Tensor] = None

    def create_child(self, token_id: int) -> 'TrieNode':
        assert token_id not in self.children, f"Child node for token {token_id} already exists"
        self.children[token_id] = TrieNode(parent=self)
        return self.children[token_id]


class Trie:
    def __init__(self):
        self.root = TrieNode()
