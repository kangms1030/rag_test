"""평가셋 로드/생성/저장을 한 곳에서 관리.

세 가지 source를 명확히 구분한다:
- human: 사람이 만든 질문 (카탈로그 `대표 예상 질문` 컬럼 시드 + 무관 질문 필러)
- synthetic: LLM이 생성한 질문 (human 시드가 부족할 때만 보충)
- sample_qa: `QA_샘플파일_100개.xlsx`에서 임포트한 실사용자 장애 Q&A

human 평가셋만 `expected_documents`가 신뢰 가능하므로 doc_recall/doc_mrr는 human으로만
집계한다 (evaluate.py). sample_qa는 `expected_documents`가 항상 비어 있어 해당 지표가 null이 된다.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .catalog import load_catalog, match_catalog_to_pdfs
from .config import Config
from .models import Backend, LLMJsonError

logger = logging.getLogger(__name__)

VALID_SOURCES = ("human", "synthetic", "sample_qa")
VALID_QUESTION_TYPES = ("procedure", "fact", "table", "diagram", "troubleshooting", "irrelevant")

PROMPT_GEN_QUESTION = """아래는 한 문서에 대한 카탈로그 정보다.

제목: {title}
설명: {description}
키워드: {keyword}

이 문서의 내용을 실제로 알아야 답할 수 있는, 사용자가 물어볼 법한 구체적인 질문을
{n}개 만들어줘. 너무 일반적이거나 카탈로그 정보만으로 답할 수 있는 질문은 안 된다.

반드시 아래 JSON 형식으로만 답해줘. 다른 텍스트는 쓰지 마.
{{"questions": ["질문1", "질문2"]}}
"""

# 카탈로그 문서와 무관한, 명백히 도메인 밖인 질문 (거절 능력 측정용 필러).
_IRRELEVANT_FILLER_QUESTIONS = [
    "김치찌개 맛있게 끓이는 방법 알려줘",
    "오늘 서울 날씨는 어때?",
    "요즘 인기 있는 영화 추천해줘",
    "삼성전자 주식 전망이 어떻게 되나요?",
]

_LEADING_NUM_RE = re.compile(r"^\s*\d+[\.\)]\s*")


def _split_sample_questions(raw: str) -> list[str]:
    """`대표 예상 질문` 셀 텍스트를 줄 단위로 쪼개고, 각 줄의 선행 번호("2. ")를 제거한다.

    이전 구현은 `re.split`의 `^`가 MULTILINE 없이는 문자열 맨 앞에서만 매칭된다는 점을
    놓쳐 첫 줄만 번호가 지워지고 2번째 줄부터는 "2. 전국 시도..."처럼 번호가 남았다.
    """
    lines = re.split(r"[\n;]", raw)
    out: list[str] = []
    for line in lines:
        line = _LEADING_NUM_RE.sub("", line).strip(" -·\t")
        if line:
            out.append(line)
    return out


def _make_item(
    *,
    id_: str,
    question: str,
    expected_answer: str = "",
    expected_documents: list[str] | None = None,
    expected_pages: list[int] | None = None,
    expected_answer_keywords: list[str] | None = None,
    question_type: str,
    category: str = "",
    source: str,
) -> dict[str, Any]:
    if question_type not in VALID_QUESTION_TYPES:
        raise ValueError(f"question_type은 {VALID_QUESTION_TYPES} 중 하나여야 함: {question_type!r}")
    if source not in VALID_SOURCES:
        raise ValueError(f"source는 {VALID_SOURCES} 중 하나여야 함: {source!r}")
    return {
        "id": id_,
        "question": question,
        "expected_answer": expected_answer,
        "expected_documents": expected_documents or [],
        "expected_pages": expected_pages or [],
        "expected_answer_keywords": expected_answer_keywords or [],
        "question_type": question_type,
        "category": category,
        "source": source,
    }


def build_human_eval(config: Config, *, target_count: int = 20, include_irrelevant: int = 4) -> list[dict]:
    """카탈로그 `대표 예상 질문`에서 문서당 1~2개 + 무관 질문 필러로 human 평가셋을 만든다.

    LLM을 호출하지 않는다 (100% 결정론적, 재현 가능). 문서 수가 목표보다 많으면
    한 문서에 못 미치는 예산이 배정될 수 있어 그 경우 문서 수만큼만 뽑는다.
    """
    rows = load_catalog(config)
    match_catalog_to_pdfs(rows, config.documents_dir)
    matched_rows = [r for r in rows if r.matched_file_path]

    doc_budget = max(0, target_count - include_irrelevant)
    n_docs = len(matched_rows)
    if n_docs == 0:
        return []

    base_per_doc = doc_budget // n_docs
    remainder = doc_budget % n_docs

    items: list[dict] = []
    qid = 0
    for i, row in enumerate(matched_rows):
        n_for_this_doc = base_per_doc + (1 if i < remainder else 0)
        if n_for_this_doc <= 0:
            continue
        document_name = Path(row.matched_file_path).name
        seed_raw = row.columns.get("sample_questions", "")
        questions = _split_sample_questions(seed_raw)[:n_for_this_doc] if seed_raw else []
        for q in questions:
            qid += 1
            items.append(
                _make_item(
                    id_=f"human_{qid:03d}",
                    question=q,
                    expected_documents=[document_name],
                    question_type="fact",
                    category=row.columns.get("theme", ""),
                    source="human",
                )
            )

    for i, q in enumerate(_IRRELEVANT_FILLER_QUESTIONS[:include_irrelevant], start=1):
        items.append(
            _make_item(
                id_=f"human_irr_{i:03d}",
                question=q,
                question_type="irrelevant",
                source="human",
            )
        )
    return items


def build_synthetic_eval(config: Config, backend: Backend, *, n_per_doc: int = 2) -> list[dict]:
    """human 시드(`대표 예상 질문`)가 `n_per_doc`보다 적은 문서만 LLM으로 보충 생성.

    생성된 항목은 전부 `source: "synthetic"`으로 표시한다 (사람이 만든 human 평가셋과
    절대 섞어 결론 내지 않기 위함, evaluate.py의 human/synthetic 분리 집계 참조).
    """
    rows = load_catalog(config)
    match_catalog_to_pdfs(rows, config.documents_dir)

    items: list[dict] = []
    qid = 0
    for row in rows:
        if not row.matched_file_path:
            continue
        title = row.columns.get("title", "")
        document_name = Path(row.matched_file_path).name

        seed_raw = row.columns.get("sample_questions", "")
        seed_questions = _split_sample_questions(seed_raw)[:n_per_doc] if seed_raw else []
        need = n_per_doc - len(seed_questions)
        generated: list[str] = []
        if need > 0:
            prompt = PROMPT_GEN_QUESTION.format(
                title=title,
                description=row.columns.get("description", ""),
                keyword=row.columns.get("keyword", ""),
                n=need,
            )
            try:
                result = backend.chat_json(prompt)
                generated = [q for q in result.get("questions", []) if isinstance(q, str) and q.strip()][:need]
            except LLMJsonError as e:
                logger.warning("%s: synthetic 질문 생성 실패: %s", document_name, e)

        for q in seed_questions:
            qid += 1
            items.append(
                _make_item(
                    id_=f"synth_{qid:03d}",
                    question=q,
                    expected_documents=[document_name],
                    question_type="fact",
                    category=row.columns.get("theme", ""),
                    source="human",
                )
            )
        for q in generated:
            qid += 1
            items.append(
                _make_item(
                    id_=f"synth_{qid:03d}",
                    question=q,
                    expected_documents=[document_name],
                    question_type="fact",
                    category=row.columns.get("theme", ""),
                    source="synthetic",
                )
            )
    return items


_WORD_RE = re.compile(r"[가-힣]{2,}|[A-Za-z0-9]{2,}")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _extract_keywords(question: str, answer: str, doc_freq: dict[str, int], *, top_k: int = 8) -> list[str]:
    """정답 답변에서 결정론적으로 키워드를 추출한다 (LLM 미사용, 재현 가능).

    문서빈도(`doc_freq`, 전체 QA셋 기준)가 낮을수록 변별력 있는 토큰으로 보고 우선한다.
    질문에도 등장하는 토큰은 관련성이 높다고 보고 함께 포함한다.
    """
    answer_words = _words(answer)
    unique_answer_words = list(dict.fromkeys(answer_words))
    ranked = sorted(unique_answer_words, key=lambda w: (doc_freq.get(w, 0), unique_answer_words.index(w)))
    top = ranked[:top_k]
    q_words = set(_words(question))
    shared = [w for w in unique_answer_words if w in q_words]
    return list(dict.fromkeys(top + shared))


def import_qa_xlsx(qa_file: str | Path, *, sheet: str = "장애 Q&A 질답쌍") -> list[dict]:
    """`QA_샘플파일_100개.xlsx`를 sample_qa 평가셋으로 변환.

    expected_documents/expected_pages는 원본에 없으므로 항상 빈 배열로 둔다 — 이 평가셋으로
    doc_recall/page_recall을 직접 재지 않는다 (evaluate.py가 null 처리).
    """
    import pandas as pd

    df = pd.read_excel(qa_file, sheet_name=sheet, dtype=object)
    df = df.dropna(subset=["질문", "답변"], how="any")

    answers = [str(a).strip() for a in df["답변"].tolist()]
    doc_freq: dict[str, int] = {}
    for a in answers:
        for w in set(_words(a)):
            doc_freq[w] = doc_freq.get(w, 0) + 1

    items: list[dict] = []
    for _, row in df.iterrows():
        no = row.get("No")
        question = str(row["질문"]).strip()
        answer = str(row["답변"]).strip()
        category = str(row.get("장애 유형", "")).strip() if not pd.isna(row.get("장애 유형")) else ""
        qid = f"qa_{int(no):03d}" if no is not None and not pd.isna(no) else f"qa_{len(items) + 1:03d}"
        items.append(
            _make_item(
                id_=qid,
                question=question,
                expected_answer=answer,
                expected_answer_keywords=_extract_keywords(question, answer, doc_freq),
                question_type="troubleshooting",
                category=category,
                source="sample_qa",
            )
        )
    return items


def load_eval_set(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if item.get("source") not in VALID_SOURCES:
            raise ValueError(f"eval item {item.get('id')}: source는 {VALID_SOURCES} 중 하나여야 함")
    return items


def save_eval_set(items: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    counts: dict[str, int] = {}
    for i in items:
        counts[i["source"]] = counts.get(i["source"], 0) + 1
    logger.info("평가셋 저장: %s (%s, 총 %d개)", out_path, counts, len(items))
