"""Memory-biased priority search on KubeGraph (Section 3.2.2, Eq 5)."""

from __future__ import annotations
import heapq
import networkx as nx
from loguru import logger

from epmn.memory import EpisodicMemory, PatternAbstraction


def prior_score(node: str, memories: list) -> float:
    """prior(M*, p) = max_{m∈M*} overlap(p, Path(m))  — Eq 5."""
    canonical_tokens: set[str] = set()
    for m in memories:
        if isinstance(m, PatternAbstraction):
            for s in m.canonical_symptoms:
                canonical_tokens.update(s.lower().split())
        elif isinstance(m, EpisodicMemory):
            for s in m.symptoms:
                canonical_tokens.update(s.lower().split())
    node_tokens = set(node.lower().replace("_", " ").split())
    if not canonical_tokens:
        return 0.0
    return len(node_tokens & canonical_tokens) / (len(node_tokens) + 1e-9)


def path_score(G: nx.DiGraph, path: list[str]) -> float:
    """score_path: average edge weight along path."""
    if len(path) < 2:
        return 0.0
    weights = [G[u][v].get("weight", 0.5) for u, v in zip(path[:-1], path[1:])]
    return sum(weights) / len(weights)


def novelty_score(node: str, visited: set[str]) -> float:
    return 0.0 if node in visited else 1.0


def memory_biased_search(
    G: nx.DiGraph,
    memories: list,
    max_hops: int = 3,
    alpha: tuple[float, float, float] = (0.5, 0.3, 0.2),
    top_n_chains: int = 5,
) -> list[list[str]]:
    """
    Extract top causal chains G* = {c1,...,cn} via priority search (Eq 5).
    priority(p) = α1·prior(M*,p) + α2·score_path(p) + α3·novelty(p)
    Returns list of node-path lists.
    """
    α1, α2, α3 = alpha

    # seed nodes: any graph node that overlaps with memory symptoms
    seed_nodes: list[str] = []
    for node in G.nodes:
        if prior_score(node, memories) > 0.1:
            seed_nodes.append(node)

    if not seed_nodes:
        # fall back: all fault-type nodes
        seed_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "fault"]

    logger.debug(f"[KubeGraph] seeds={seed_nodes}")

    # max-heap (negate priority for heapq min-heap)
    visited_paths: set[tuple[str, ...]] = set()
    visited_nodes: set[str] = set()
    heap: list[tuple[float, list[str]]] = []

    for s in seed_nodes:
        pr = prior_score(s, memories)
        heapq.heappush(heap, (-pr, [s]))

    causal_chains: list[list[str]] = []

    while heap and len(causal_chains) < top_n_chains * 3:
        neg_prio, path = heapq.heappop(heap)
        current = path[-1]
        path_key = tuple(path)

        if path_key in visited_paths:
            continue
        visited_paths.add(path_key)

        # Emit chain if it ends at a resolution node or hits max hops
        if (len(path) >= 2 and
                G.nodes[current].get("type") in {"resolution", "cause"} and
                len(path) >= 2):
            causal_chains.append(path)
            if len(causal_chains) >= top_n_chains:
                break

        if len(path) >= max_hops + 1:
            continue

        for neighbor in G.successors(current):
            nov = novelty_score(neighbor, visited_nodes)
            new_path = path + [neighbor]
            ps = path_score(G, new_path)
            pr = prior_score(neighbor, memories)
            prio = α1 * pr + α2 * ps + α3 * nov
            heapq.heappush(heap, (-prio, new_path))
            visited_nodes.add(neighbor)

    logger.info(f"[KubeGraph] found {len(causal_chains)} causal chains")
    return causal_chains[:top_n_chains]


def hints_for_epmn(G: nx.DiGraph, memories: list, top_k: int = 5) -> list[str]:
    """
    Hints = {v ∈ V_G | ∃m∈TopK_hint(Q): v ∈ Path(m)}  — Eq 16.
    Returns node names that appear in memory-relevant paths.
    """
    chains = memory_biased_search(G, memories, top_n_chains=top_k)
    hint_nodes: list[str] = []
    seen: set[str] = set()
    for chain in chains:
        for node in chain:
            if node not in seen:
                hint_nodes.append(node)
                seen.add(node)
    return hint_nodes
