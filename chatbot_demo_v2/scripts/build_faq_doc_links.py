"""FAQ 근거링크 생성 (데이터 격차 D1/D2).

faq.json 의 source_files("문서명 N쪽" / "N~M쪽" / "문서명 9~16쪽")를 파싱해 코퍼스(page_store)의
문서와 매칭하고 data/faq_doc_links.json 을 만든다. 코퍼스에 없는 참조 문서는
runtime/reports/corpus_gap.md 로 리포트한다(원본 PDF 확보·증분색인은 사용자 결정 사항).

실행:
    <intern_chatbot python> -m chatbot_demo_v2.scripts.build_faq_doc_links
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "ragcore"))

# "…문서명… 12쪽" / "9~10쪽" / "9-16쪽" (문서명은 없을 수 있음 → 직전 문서 승계)
_REF_RE = re.compile(r"^(?P<doc>.*?)\s*(?P<a>\d+)\s*(?:[~\-–]\s*(?P<b>\d+))?\s*쪽")
_MAX_PAGES_PER_REF = 12          # "9~16쪽" 같은 범위 폭주 방지


def _norm_doc(s: str) -> str:
    """문서명 정규화 — 매칭 키. 기호/공백/확장자 제거 후 소문자."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\.(pdf|docx?|hwpx?|pptx?)\b", "", s, flags=re.I)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s.lower()


def parse_source_files(source_files: list[str]) -> list[tuple[str, list[int]]]:
    """['문서명 7쪽', '28쪽', '가이드 9~10쪽'] → [(문서명,[7]), (문서명,[28]), (가이드,[9,10])].

    문서명이 비면 직전 참조의 문서명을 승계한다(엑셀 표기 관행).
    """
    out: list[tuple[str, list[int]]] = []
    last_doc = ""
    for raw in source_files or []:
        for part in re.split(r"\s*,\s*", str(raw).strip()):
            if not part:
                continue
            m = _REF_RE.match(part)
            if not m:
                # 쪽 표기가 없는 순수 문서명 참조
                doc = part.strip()
                if doc:
                    last_doc = doc
                    out.append((doc, []))
                continue
            doc = (m.group("doc") or "").strip() or last_doc
            if doc:
                last_doc = doc
            a = int(m.group("a"))
            b = int(m.group("b")) if m.group("b") else a
            if b < a:
                a, b = b, a
            pages = list(range(a, min(b, a + _MAX_PAGES_PER_REF - 1) + 1))
            out.append((doc, pages))
    return out


def build_corpus_index() -> dict[str, dict]:
    """page_store 에서 정규화문서명 → {document_name, doc_slug, pages:{page: image_rel}} 구축.

    image_rel 은 파싱 캐시 루트(cfg.source_parsed) 기준 상대경로다. 이렇게 저장해 두면
    **런타임(FAQ 경로)에서는 ragcore/page_store 를 로드하지 않고** 파일만 복사하면 된다.
    """
    from rag3.answer import resolve_cached_path
    from rag3.config import load_config
    from rag3.page_store import load_page_store

    cfg = load_config(PKG / "ragcore" / "rag3" / "config.yaml")
    parsed_root = Path(cfg.source_parsed)
    store = load_page_store(cfg)
    idx: dict[str, dict] = {}
    for rec in store.values():
        meta = rec.get("meta", {})
        name = meta.get("document_name") or ""
        slug = meta.get("doc_slug") or ""
        try:
            page = int(meta.get("page_number"))
        except (TypeError, ValueError):
            continue
        key = _norm_doc(name)
        entry = idx.setdefault(key, {"document_name": name, "doc_slug": slug, "pages": {}})
        if page in entry["pages"]:
            continue
        rel = None
        img = resolve_cached_path(meta.get("page_image_path", ""), cfg)
        if img is not None:
            try:
                rel = str(Path(img).resolve().relative_to(parsed_root.resolve())).replace("\\", "/")
            except ValueError:
                rel = None
        entry["pages"][page] = rel
    return idx


#: 유사매칭 채택 조건 — 최고점 하한 + 2위와의 격차(애매하면 거절).
#  FAQ 유사도 매처(threshold+margin)와 같은 원칙. 실측 분리도:
#    동일문서 '스쿨넷서비스…가이드(2018.12)' → 77.6 / 격차 23.0  (채택)
#    부재문서 ★05__FAQ.docx 33.3·격차 4.8, ★홈페이지_쳇봇문의 18.2·격차 1.5,
#            학내전산망 설치기준 47.1·격차 5.0, 세종 가이드 48.0·격차 1.3 (전부 거절)
_FUZZY_FLOOR = 75
_FUZZY_MARGIN = 15


def match_doc(doc_name: str, corpus: dict[str, dict]) -> dict | None:
    """정규화 후 완전일치 → 부분포함 → 유사매칭(partial_ratio) 순.

    유사매칭이 필요한 이유(실측): FAQ 표기가 코퍼스 파일명의 축약/변형인 경우가 있어
    포함관계가 성립하지 않는다.
      - "스쿨넷서비스 학내망 구축 및 운영·관리 가이드(2018.12)"
        vs 코퍼스 "8-1. 스쿨넷서비스 학내망 구축 및 운영관리 **개선을 위한** 가이드.pdf"
      - "23년 학교 유무선 운영 관리 안내서**(학내망)**"
        vs 코퍼스 "★23년 학교 유무선 운영·관리 안내서**_최종**.pdf"
    반환값에 match_method 를 실어 감사 가능하게 한다.
    """
    key = _norm_doc(doc_name)
    if not key:
        return None
    if key in corpus:
        return {**corpus[key], "match_method": "exact"}

    # 포함관계 — 동점(여러 문서에 똑같이 포함)이면 애매하므로 채택하지 않는다.
    hits: list[tuple[int, dict]] = []
    for ckey, entry in corpus.items():
        if len(ckey) < 6 or len(key) < 6:
            continue
        if key in ckey or ckey in key:
            hits.append((min(len(key), len(ckey)), entry))
    if hits:
        hits.sort(key=lambda t: t[0], reverse=True)
        top = hits[0][0]
        if len(hits) == 1 or hits[1][0] < top:
            return {**hits[0][1], "match_method": "contains"}
        return None      # 최상위 동점 → 어느 문서인지 확정 불가

    # 유사매칭 — 최고점이 하한 이상 ∧ 2위와 충분히 벌어졌을 때만(애매하면 거절)
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None
    if len(key) < 6:
        return None
    scored = sorted(
        ((fuzz.partial_ratio(key, ckey), entry)
         for ckey, entry in corpus.items() if len(ckey) >= 6),
        key=lambda t: t[0], reverse=True,
    )
    if not scored:
        return None
    best_score, best_entry = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= _FUZZY_FLOOR and (best_score - second_score) >= _FUZZY_MARGIN:
        return {**best_entry,
                "match_method": f"fuzzy({best_score:.0f}, margin {best_score - second_score:.0f})"}
    return None


def main() -> int:
    faq_path = PKG / "data" / "faq.json"
    out_path = PKG / "data" / "faq_doc_links.json"
    gap_path = PKG / "runtime" / "reports" / "corpus_gap.md"

    faq = json.loads(faq_path.read_text(encoding="utf-8"))
    entries = faq.get("entries", [])
    corpus = build_corpus_index()
    print(f"코퍼스 문서 {len(corpus)}종, FAQ {len(entries)}행")

    links: dict[str, list[dict]] = {}
    unmatched: dict[str, list[str]] = defaultdict(list)   # 문서명 → [faq_id]
    out_of_range: list[str] = []                          # 문서는 맞지만 페이지가 범위 밖
    fuzzy_matches: dict[tuple[str, str, str], int] = {}    # 유사매칭 감사용
    n_linked = 0        # 쪽번호까지 확보 → 근거 이미지 표시 가능
    n_doc_only = 0      # 문서만 확보(쪽번호 미인용) → 출처 문서명만 표시

    for e in entries:
        refs = parse_source_files(e.get("source_files") or [])
        per_doc: dict[str, dict] = {}
        for doc_name, pages in refs:
            hit = match_doc(doc_name, corpus)
            if hit is None:
                if doc_name:
                    unmatched[doc_name].append(e["id"])
                continue
            method = hit.get("match_method", "exact")
            if method.startswith("fuzzy"):
                fuzzy_matches.setdefault((doc_name, hit["document_name"], method), 0)
                fuzzy_matches[(doc_name, hit["document_name"], method)] += 1
            slot = per_doc.setdefault(hit["doc_slug"], {
                "doc_slug": hit["doc_slug"],
                "document_name": hit["document_name"],
                "match_method": method,
                "pages": [],
            })
            valid = [p for p in pages if p in hit["pages"]]
            if pages and not valid:
                out_of_range.append(
                    f"{e['id']}: '{doc_name}' {pages} (문서 페이지 1~{max(hit['pages'])})"
                )
            for p in valid:
                if not any(x["page"] == p for x in slot["pages"]):
                    slot["pages"].append({"page": p, "image_rel": hit["pages"][p]})
        # 쪽번호가 없는 참조도 **문서 단위 출처**로 남긴다(pages=[]).
        # 실측: FAQ 236행 중 199행이 쪽번호 없이 문서명만 인용한다 → 이미지는 못 붙여도
        # "이 답변의 근거 문서" 는 보여줄 수 있다.
        rows = list(per_doc.values())
        if rows:
            for r in rows:
                r["pages"].sort(key=lambda x: x["page"])
            links[e["id"]] = rows
            if any(r["pages"] for r in rows):
                n_linked += 1
            else:
                n_doc_only += 1

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": {"faq_total": len(entries),
                  "faq_linked_pages": n_linked,        # 근거 이미지 표시 가능
                  "faq_linked_docs_only": n_doc_only,  # 출처 문서명만 표시
                  "unmatched_docs": len(unmatched)},
        "links": links,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"faq_doc_links.json 생성: 근거페이지 {n_linked}행 + 출처문서만 {n_doc_only}행 "
          f"= {n_linked + n_doc_only}/{len(entries)}")

    # --- 갭 리포트 ---
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 코퍼스 커버리지 갭 리포트",
        "",
        f"생성: {out['generated_at']}",
        "",
        f"- FAQ 총 {len(entries)}행 중 **{n_linked}행**에 근거 **페이지**(이미지 표시 가능)를,",
        f"  **{n_doc_only}행**에 근거 **문서명만** 연결했습니다(FAQ가 쪽번호를 인용하지 않음).",
        f"- 코퍼스(13문서, 969페이지)에 **없는** 참조 문서: **{len(unmatched)}종**",
        "",
        "> **핵심**: 근거 이미지가 34행에 그치는 주된 원인은 코퍼스 격차가 아니라",
        "> **FAQ 원본의 인용 방식**입니다. 236행 중 199행이 쪽번호 없이 문서명만 인용합니다.",
        "> 따라서 아래 미매칭 문서를 추가 색인해도 **근거 이미지는 늘지 않습니다**",
        "> (해당 참조 94건 전부 쪽번호 없음). 색인 추가의 가치는 'RAG 검색 대상 확대'로 별개입니다.",
        "",
        "## 코퍼스에 없는 참조 문서",
        "",
        "| 참조 문서명(FAQ 표기) | 참조 FAQ 행 수 | 예시 FAQ id |",
        "|---|---:|---|",
    ]
    for doc, ids in sorted(unmatched.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {doc} | {len(ids)} | {', '.join(ids[:3])} |")
    lines += [
        "",
        "## 유사매칭으로 연결된 참조 (검토 권장)",
        "",
        "FAQ 표기가 코퍼스 파일명의 축약/변형이라 포함관계로는 못 찾고 유사도로 연결한 건입니다.",
        "",
        "| FAQ 표기 | 연결된 코퍼스 문서 | 유사도 | 참조 수 |",
        "|---|---|---|---:|",
        *[f"| {a} | {b} | {m} | {n} |"
          for (a, b, m), n in sorted(fuzzy_matches.items(), key=lambda kv: -kv[1])],
        "",
        "## 문서는 매칭됐으나 인용 페이지가 문서 범위를 벗어난 참조",
        "",
        f"총 {len(out_of_range)}건 — 인용된 판본과 코퍼스 판본이 다를 수 있습니다.",
        "",
    ]
    lines += [f"- {row}" for row in out_of_range[:20]]
    lines += [
        "",
        "## 조치 옵션(사용자 결정 필요)",
        "",
        "1. 위 문서의 원본 PDF를 확보해 `ragcore/rag3/add_doc.py` 로 증분 색인 → RAG가 해당 FAQ 근거도 검색 가능해집니다.",
        "2. 확보가 어려우면 현행 유지 — 해당 FAQ는 **저장된 모범답변으로는 정상 응답**하며 근거 이미지만 표시되지 않습니다.",
        "",
        "> 참고: 이 갭은 답변 품질 문제가 아니라 '근거 이미지/RAG 검색 대상' 범위 문제입니다.",
    ]
    gap_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"corpus_gap.md 생성: 미매칭 문서 {len(unmatched)}종")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
