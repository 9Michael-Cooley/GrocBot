"""Tree attention for speculative decoding. UNFINISHED.

Instead of one linear draft of k tokens, the draft model proposes a tree of
candidate continuations and the target verifies all branches in a single pass
with a block-diagonal-ish mask. Acceptance goes up because you are no longer
betting on one sequence.

Status: mask construction works and is tested by hand below. Nothing calls it.
The verify path in inference/speculative.py assumes a flat list of drafted
tokens and would need to walk the tree to attribute acceptances, which is the
part that was never written.

Branch was ~6 weeks stale at extraction. Do not assume any of this matches the
current SpeculativeDecoder interface, because it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    token_id: int
    parent: int | None
    depth: int
    prob: float = 0.0
    children: list[int] = field(default_factory=list)


class DraftTree:
    """Candidate continuations as a tree. Node 0 is the root (last real token)."""

    def __init__(self, root_token: int):
        self.nodes: list[TreeNode] = [TreeNode(root_token, None, 0, 1.0)]

    def add(self, parent: int, token_id: int, prob: float) -> int:
        if parent >= len(self.nodes):
            raise IndexError(f"no node {parent}")
        idx = len(self.nodes)
        self.nodes.append(TreeNode(token_id, parent, self.nodes[parent].depth + 1, prob))
        self.nodes[parent].children.append(idx)
        return idx

    def path_to(self, idx: int) -> list[int]:
        path: list[int] = []
        while idx is not None:
            path.append(idx)
            idx = self.nodes[idx].parent  # type: ignore[assignment]
        return list(reversed(path))

    def leaves(self) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if not n.children]

    @property
    def size(self) -> int:
        return len(self.nodes)

    def attention_mask(self) -> list[list[int]]:
        """mask[i][j] == 1 iff j is an ancestor of i (or i itself).

        This is the whole trick: every node attends to its own path and nothing
        else, so sibling branches cannot see each other and one forward pass
        scores every branch independently.
        """
        n = self.size
        mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for ancestor in self.path_to(i):
                mask[i][ancestor] = 1
        return mask

    def position_ids(self) -> list[int]:
        """Depth is the position. Siblings share one — that's intended, and it is
        also why this cannot use the standard RoPE cache path."""
        return [node.depth for node in self.nodes]

    def cumulative_probs(self) -> list[float]:
        out = [0.0] * self.size
        out[0] = 1.0
        for i in range(1, self.size):
            parent = self.nodes[i].parent
            out[i] = out[parent] * self.nodes[i].prob  # type: ignore[index]
        return out

    def prune(self, max_nodes: int) -> DraftTree:
        """Keep the highest-probability nodes.

        BUG: prunes by cumulative probability without checking that a kept node's
        parent is also kept, so it can orphan a subtree and produce a mask with
        gaps. Needs to prune top-down. This is why nothing calls it.
        """
        scores = self.cumulative_probs()
        keep = sorted(range(self.size), key=lambda i: scores[i], reverse=True)[:max_nodes]
        keep_set = set(keep) | {0}

        pruned = DraftTree(self.nodes[0].token_id)
        remap = {0: 0}
        for i in sorted(keep_set - {0}):
            parent = self.nodes[i].parent
            if parent not in remap:
                continue  # <-- the orphan case, silently dropped
            remap[i] = pruned.add(remap[parent], self.nodes[i].token_id, self.nodes[i].prob)
        return pruned


def expand_greedy(tree: DraftTree, draft_fn, width: int, depth: int) -> DraftTree:
    """Breadth-first expansion. `draft_fn(path) -> [(token, prob), ...]`."""
    frontier = [0]
    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            path = [tree.nodes[i].token_id for i in tree.path_to(node)]
            for token, prob in draft_fn(path)[:width]:
                next_frontier.append(tree.add(node, token, prob))
        frontier = next_frontier
        if not frontier:
            break
    return tree
