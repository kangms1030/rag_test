"""시나리오 트리 탐색(결정론적)."""

from __future__ import annotations

from .models import ScenarioNode


class InvalidActionError(Exception):
    """알 수 없는 node_id/option_id 등 잘못된 시나리오 액션."""


class ScenarioTree:
    def __init__(self, root_node_id: str, nodes: dict[str, ScenarioNode]):
        self.root_node_id = root_node_id
        self._nodes = dict(nodes)

    @property
    def nodes(self) -> dict[str, ScenarioNode]:
        return dict(self._nodes)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> ScenarioNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise InvalidActionError(f"존재하지 않는 노드: {node_id}")
        return node

    def root(self) -> ScenarioNode:
        return self._nodes[self.root_node_id]

    def resolve_option(self, node_id: str, option_id: str) -> ScenarioNode:
        """현재 노드에서 option_id 를 눌렀을 때 이동할 다음 노드."""
        node = self.get_node(node_id)
        for opt in node.options:
            if opt.option_id == option_id:
                return self.get_node(opt.next_node_id)
        raise InvalidActionError(
            f"노드 {node_id} 에 option_id '{option_id}' 없음"
        )

    def options_payload(self, node: ScenarioNode) -> list[dict]:
        """노드의 선택지를 프론트 버튼 payload 목록으로."""
        return [opt.to_button(node.scenario_id, node.node_id) for opt in node.options]

    def root_payload(self) -> dict:
        node = self.root()
        return {
            "scenario_id": node.scenario_id,
            "node_id": node.node_id,
            "text": node.text,
            "options": self.options_payload(node),
        }
