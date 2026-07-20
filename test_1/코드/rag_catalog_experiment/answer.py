"""VLM 답변 생성 + 근거 crop/highlight 저장.

`vlm_test_v2.ipynb`의 PROMPT_ANSWER를 재사용하되, 그 노트북의 실제 버그
(evidence bbox가 항상 첫 번째 이미지에 귀속되던 문제)를 고치기 위해 프롬프트에
이미지 매니페스트를 넣고 각 evidence가 `image_index`로 자신이 나온 이미지를
명시하도록 강제했다.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from . import metrics
from .config import Config
from .imaging import crop_image, highlight_boxes, is_valid_bbox, save_jpeg
from .models import Backend, LLMJsonError
from .ocr import fill_evidence_text
from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

PROMPT_ANSWER = """다음 질문에 답해줘.

질문: {question}

아래는 참고할 수 있는 이미지 목록이다:
{image_manifest}

반드시 아래 JSON 형식으로만 답해줘. 다른 텍스트는 쓰지 마.
{{
  "answer": "질문에 대한 답변 텍스트",
  "evidence": [
    {{
      "image_index": 1,
      "description": "이 영역이 근거인 이유",
      "confidence": 0.9,
      "x1": 0.0,
      "y1": 0.0,
      "x2": 1.0,
      "y2": 1.0
    }}
  ]
}}

규칙:
- image_index는 위 이미지 목록의 번호(1부터 시작)와 정확히 일치해야 한다 (근거가 어느 이미지에서 나왔는지 반드시 명시)
- evidence의 bbox는 해당 이미지 내 비율(0.0~1.0)로 표시, 왼쪽위(x1,y1)~오른쪽아래(x2,y2)
- confidence는 이 근거가 답변을 얼마나 확실하게 뒷받침하는지 0.0~1.0
- 근거 영역이 여러 개면 배열에 모두 포함
- 이미지에 없는 내용은 추측하지 마
- 근거를 찾지 못하면 answer에 "선택된 문서에서 확인할 수 없습니다"라고 쓰고 evidence는 빈 배열로
"""

PROMPT_VERIFY = """다음은 질문에 대한 답변과 그 근거다. 답변이 근거만으로 뒷받침되는지 검증해줘.

질문: {question}
답변: {answer}

근거:
{evidence_text}

반드시 아래 JSON 형식으로만 답해줘. 다른 텍스트는 쓰지 마.
{{
  "is_answer_supported": true,
  "unsupported_claims": ["근거로 확인되지 않는 주장들"],
  "notes": "판단 이유"
}}
"""

# visual_description(텍스트)만 보고 검증하면 VLM이 표/숫자를 잘못 읽었을 때 검증도 같이
# 틀린다(REPORT.md A1/A4 실측: 27,054->27,004 오독을 텍스트 검증이 못 잡음). 숫자 주장은
# crop 이미지를 다시 보여주며 재검증한다(config.verify_with_crop_image).
PROMPT_VERIFY_NUMERIC = """다음은 답변에 포함된 숫자/수치 주장이다. 첨부된 근거 확대 이미지를 보고
이 숫자들이 이미지 내용과 정확히 일치하는지 하나씩 대조해줘. 이미지에 없는 숫자는 확인 불가로 표시해.

질문: {question}
답변: {answer}
확인할 숫자 주장: {claims}

반드시 아래 JSON 형식으로만 답해줘. 다른 텍스트는 쓰지 마.
{{
  "numeric_verification_notes": "이미지와 대조한 결과를 숫자별로 설명",
  "unsupported_numeric_claims": ["이미지로 확인되지 않거나 이미지 내용과 다른 숫자들"]
}}
"""

_NO_DOC_ANSWER = "선택된 문서에서 확인 불가"

# 쉼표로 묶인 큰 수(83,375) / 소수 / %(단독) / 한글 단위가 붙은 수(3위, 12개, 5배 등).
# 페이지 번호 같은 우연한 숫자를 줄이려고 "단위 없는 순수 정수"는 제외한다.
_NUMERIC_CLAIM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d+%|\d+(?:위|등|개|명|대|건|배|년|월|일)")


def extract_numeric_claims(answer_text: str) -> list[str]:
    """답변 텍스트에서 숫자 주장을 결정론적으로 추출(순서 보존, 중복 제거)."""
    return list(dict.fromkeys(_NUMERIC_CLAIM_RE.findall(answer_text)))


def _format_manifest(images: list[dict]) -> str:
    return "\n".join(f"[이미지 {i}] 문서: {img['document_name']} / 페이지: {img['page_number']}" for i, img in enumerate(images, start=1))


def generate_answer(question: str, retrieval: RetrievalResult, backend: Backend, config: Config) -> dict[str, Any]:
    """{"final_answer", "raw_evidence", "images_used"} 반환. VLM을 호출하지 않는 조기 종료 케이스 포함."""
    if not retrieval.selected_documents or not retrieval.selected_pages:
        reason = "카탈로그에서 관련 문서를 찾지 못함" if not retrieval.selected_documents else "선정된 문서에서 관련 페이지를 찾지 못함"
        logger.info("답변 생성 조기 종료: %s", reason)
        return {"final_answer": _NO_DOC_ANSWER, "raw_evidence": [], "images_used": [], "skip_reason": reason}

    images = retrieval.selected_pages[: config.max_images_per_call]
    manifest_text = _format_manifest(images)
    prompt = PROMPT_ANSWER.format(question=question, image_manifest=manifest_text)
    image_paths = [Path(img["image_path"]) for img in images]

    try:
        result = backend.chat_vision_json(prompt, image_paths)
        metrics.record_llm("answer")
    except LLMJsonError as e:
        logger.warning("답변 생성 JSON 파싱 실패: %s", e)
        return {"final_answer": _NO_DOC_ANSWER, "raw_evidence": [], "images_used": images, "skip_reason": f"VLM JSON 파싱 실패: {e}"}

    answer_text = str(result.get("answer", "")).strip() or _NO_DOC_ANSWER
    raw_evidence = result.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []

    return {"final_answer": answer_text, "raw_evidence": raw_evidence, "images_used": images}


def build_page_evidence(
    raw_evidence: list[dict],
    images: list[dict],
    selected_documents: list[dict],
    run_id: str,
    config: Config,
) -> list[dict[str, Any]]:
    doc_slug_by_name = {d["document_name"]: d["doc_slug"] for d in selected_documents}
    file_path_by_name = {d["document_name"]: d["file_path"] for d in selected_documents}

    by_image: dict[int, list[dict]] = {}
    for ev in raw_evidence:
        idx = ev.get("image_index")
        if not isinstance(idx, int) or not (1 <= idx <= len(images)):
            logger.warning("evidence.image_index 범위 밖 또는 누락, 무시: %s", ev)
            continue
        if not is_valid_bbox(ev):
            logger.warning("evidence bbox 유효성 검사 실패, 무시: %s", ev)
            continue
        by_image.setdefault(idx, []).append(ev)

    page_evidence: list[dict[str, Any]] = []
    for idx, evs in by_image.items():
        img = images[idx - 1]
        doc_name = img["document_name"]
        page_num = img["page_number"]
        slug = doc_slug_by_name.get(doc_name, "doc")
        image_path = Path(img["image_path"])

        boxes = [{"x1": e["x1"], "y1": e["y1"], "x2": e["x2"], "y2": e["y2"]} for e in evs]
        highlighted_path = config.evidence_dir / run_id / f"{slug}_p{page_num:04d}_highlighted.jpg"
        save_jpeg(highlight_boxes(image_path, boxes), highlighted_path)

        for i, ev in enumerate(evs):
            crop_img, _ = crop_image(image_path, ev)
            crop_path = config.evidence_dir / run_id / f"{slug}_p{page_num:04d}_ev{i + 1}_crop.jpg"
            save_jpeg(crop_img, crop_path)
            confidence = ev.get("confidence", 0.5)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 0.5
            entry = {
                # final_answer 본문이 "(이미지 N)"으로 근거를 지목하므로, 그 N을 그대로 남겨야
                # JSON만 보고도 어느 근거인지 역추적할 수 있다. source_images 매니페스트와 짝을 이룬다.
                "image_index": idx,
                "document_name": doc_name,
                "page_number": page_num,
                "evidence_type": "visual",
                "evidence_text": "",
                "visual_description": str(ev.get("description", "")),
                "bbox": {"x1": ev["x1"], "y1": ev["y1"], "x2": ev["x2"], "y2": ev["y2"]},
                "crop_image_path": str(crop_path),
                "highlighted_page_path": str(highlighted_path),
                "why_relevant": str(ev.get("description", "")),
                "confidence": confidence,
            }

            rel_path = file_path_by_name.get(doc_name)
            abs_pdf_path = (config.documents_dir / rel_path) if rel_path else None
            entry["evidence_text"] = fill_evidence_text(entry, config=config, abs_pdf_path=abs_pdf_path)
            if entry["evidence_text"].strip():
                entry["evidence_type"] = "mixed"  # 시각 근거 + 원문/OCR 텍스트 둘 다 있음

            page_evidence.append(entry)
    page_evidence.sort(key=lambda e: (e["image_index"], e["page_number"]))
    return page_evidence


def verify_answer(question: str, answer_text: str, page_evidence: list[dict], backend: Backend, config: Config) -> dict[str, Any]:
    numeric_claims = extract_numeric_claims(answer_text)
    empty_result = {
        "is_answer_supported": False,
        "unsupported_claims": [],
        "numeric_claims": numeric_claims,
        "numeric_verification_notes": "",
        "notes": "",
    }
    if answer_text == _NO_DOC_ANSWER or not page_evidence:
        empty_result["notes"] = "근거(page_evidence)가 없어 검증을 건너뜀" if answer_text != _NO_DOC_ANSWER else "관련 문서/페이지를 찾지 못함"
        return empty_result

    evidence_text = "\n".join(f"- ({e['document_name']} p{e['page_number']}) {e['visual_description']}" for e in page_evidence)
    prompt = PROMPT_VERIFY.format(question=question, answer=answer_text, evidence_text=evidence_text)
    try:
        result = backend.chat_json(prompt)
        metrics.record_llm("verify")
    except LLMJsonError as e:
        logger.warning("답변 검증 실패: %s", e)
        empty_result["notes"] = f"검증 LLM 호출 실패: {e}"
        return empty_result

    result.setdefault("is_answer_supported", False)
    result.setdefault("unsupported_claims", [])
    result.setdefault("notes", "")
    result["numeric_claims"] = numeric_claims
    result["numeric_verification_notes"] = ""

    if numeric_claims and config.verify_with_crop_image:
        crop_paths = [Path(e["crop_image_path"]) for e in page_evidence[: config.verify_max_crop_images]]
        numeric_prompt = PROMPT_VERIFY_NUMERIC.format(question=question, answer=answer_text, claims=", ".join(numeric_claims))
        try:
            numeric_result = backend.chat_vision_json(numeric_prompt, crop_paths)
            metrics.record_llm("verify")
            result["numeric_verification_notes"] = str(numeric_result.get("numeric_verification_notes", ""))
            unsupported_numeric = [str(c) for c in numeric_result.get("unsupported_numeric_claims", []) if isinstance(c, (str, int, float))]
            if unsupported_numeric:
                result["unsupported_claims"] = list(result["unsupported_claims"]) + [f"[숫자] {c}" for c in unsupported_numeric]
                result["is_answer_supported"] = False
        except LLMJsonError as e:
            logger.warning("숫자 재검증 실패: %s", e)
            result["numeric_verification_notes"] = f"숫자 재검증 실패: {e}"

    return result
