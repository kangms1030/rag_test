"""faq.json / scenarios.json 로더 + 검증."""

from __future__ import annotations

import json
from pathlib import Path

from .models import FaqEntry, FaqStore, ScenarioNode, ScenarioOption
from .tree import ScenarioTree

RESERVED_OPTION_RESTART = "__restart__"


def load_faq(path: Path) -> FaqStore:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    entries: list[FaqEntry] = []
    for raw in data.get("entries", []):
        entries.append(
            FaqEntry(
                id=raw["id"],
                sheet=raw["sheet"],
                row=int(raw["row"]),
                no=raw.get("no"),
                question_type=raw.get("question_type"),
                fault_type=raw.get("fault_type"),
                question=raw["question"],
                question_normalized=raw["question_normalized"],
                answer=raw["answer"],
                source_files=list(raw.get("source_files", [])),
            )
        )
    return FaqStore(entries)


def load_scenarios(path: Path, faq: FaqStore) -> ScenarioTree:
    """시나리오 트리를 로드하고 무결성을 검증한다.

    검증:
      - root_node_id 존재
      - 모든 option.next_node_id 가 실제 노드를 가리킴
      - terminal 노드는 해석 가능한 답변(text 또는 faq_ref)을 가짐
      - option_id 는 노드 내 유일
    answer_ref 는 로드 시점에 faq.json 에서 답변 원문으로 해석한다.
    """
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    root_node_id = data.get("root_node_id")
    raw_nodes = data.get("nodes", {})
    if root_node_id not in raw_nodes:
        raise ValueError(f"root_node_id '{root_node_id}' 가 nodes 에 없음")

    nodes: dict[str, ScenarioNode] = {}
    for node_id, raw in raw_nodes.items():
        if raw.get("node_id", node_id) != node_id:
            raise ValueError(f"노드 키와 node_id 불일치: {node_id} vs {raw.get('node_id')}")

        node_type = raw.get("type", "question")
        options_raw = raw.get("options", [])
        seen_opt: set[str] = set()
        options: list[ScenarioOption] = []
        for o in options_raw:
            oid = o["option_id"]
            if oid in seen_opt:
                raise ValueError(f"노드 {node_id} 에 중복 option_id: {oid}")
            seen_opt.add(oid)
            options.append(
                ScenarioOption(
                    option_id=oid,
                    label=o["label"],
                    next_node_id=o["next_node_id"],
                )
            )

        answer_text = None
        answer_source = None
        answer_ref_sheet = None
        answer_ref_row = None
        if node_type == "terminal":
            answer = raw.get("answer")
            if not answer:
                raise ValueError(f"terminal 노드 {node_id} 에 answer 없음")
            answer_source = answer.get("source")
            if answer_source == "scenario_ppt":
                answer_text = answer.get("text")
                if not answer_text:
                    raise ValueError(f"terminal 노드 {node_id}: scenario_ppt 인데 text 없음")
            elif answer_source == "faq_ref":
                ref = answer.get("answer_ref", {})
                answer_ref_sheet = ref.get("sheet")
                answer_ref_row = ref.get("row")
                entry = faq.get_by_sheet_row(answer_ref_sheet, int(answer_ref_row))
                if entry is None:
                    raise ValueError(
                        f"terminal 노드 {node_id}: faq_ref "
                        f"{answer_ref_sheet}:{answer_ref_row} 에 해당하는 FAQ 없음"
                    )
                answer_text = entry.answer
            else:
                raise ValueError(
                    f"terminal 노드 {node_id}: 알 수 없는 answer.source '{answer_source}'"
                )

        nodes[node_id] = ScenarioNode(
            node_id=node_id,
            scenario_id=raw.get("scenario_id", node_id),
            node_type=node_type,
            text=raw.get("text"),
            options=options,
            answer_text=answer_text,
            answer_source=answer_source,
            answer_ref_sheet=answer_ref_sheet,
            answer_ref_row=answer_ref_row,
        )

    # next_node_id 참조 무결성
    for node in nodes.values():
        for opt in node.options:
            if opt.next_node_id not in nodes:
                raise ValueError(
                    f"노드 {node.node_id} 의 option '{opt.option_id}' 가 "
                    f"존재하지 않는 노드 '{opt.next_node_id}' 를 가리킴"
                )

    return ScenarioTree(root_node_id=root_node_id, nodes=nodes)
