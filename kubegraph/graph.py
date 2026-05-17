"""KubeGraph wrapper: load toy or full graph, expose traversal API."""

from __future__ import annotations
import networkx as nx
from loguru import logger

from .traversal import memory_biased_search, hints_for_epmn


class KubeGraph:
    def __init__(self, G: nx.DiGraph | None = None):
        if G is None:
            from .toy_graph import build_toy_graph
            G = build_toy_graph()
        self._G = G
        logger.info(f"[KubeGraph] loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    @classmethod
    def from_toy(cls) -> "KubeGraph":
        from .toy_graph import build_toy_graph
        return cls(build_toy_graph())

    def add_entity(self, node_id: str, node_type: str, **attrs) -> None:
        self._G.add_node(node_id, type=node_type, **attrs)

    def add_relation(self, src: str, dst: str, rel_type: str, weight: float = 0.5) -> None:
        self._G.add_edge(src, dst, type=rel_type, weight=weight)

    def memory_biased_search(
        self,
        memories: list,
        max_hops: int = 3,
        alpha: tuple[float, float, float] = (0.5, 0.3, 0.2),
        top_n: int = 5,
    ) -> list[list[str]]:
        return memory_biased_search(self._G, memories, max_hops=max_hops, alpha=alpha, top_n_chains=top_n)

    def get_hints(self, memories: list, top_k: int = 5) -> list[str]:
        return hints_for_epmn(self._G, memories, top_k=top_k)

    def neighbors(self, node: str) -> list[str]:
        return list(self._G.successors(node))

    def node_attrs(self, node: str) -> dict:
        return dict(self._G.nodes.get(node, {}))

    def stats(self) -> dict:
        return {
            "nodes": self._G.number_of_nodes(),
            "edges": self._G.number_of_edges(),
            "categories": list({d.get("category") for _, d in self._G.nodes(data=True)}),
        }
