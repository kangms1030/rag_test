# -*- coding: utf-8 -*-
"""FAQ 236행 질문을 임베딩해 data/faq_embeddings.json 으로 저장한다 (작업 3, 오프라인 1회).

왜 필요한가
    현재 scenario/matcher.py 는 rapidfuzz.fuzz.ratio(문자 편집 유사도)만 쓴다. 한국어
    패러프레이즈는 이 방식으로 임계 0.90 을 절대 넘지 못해 FAQ 236행이 사실상 사문화돼 있다
    (실측: "학교에서 새로운 ap를 설치하고 싶어" ↔ "우리 학교 와이파이는 누가 설치하고
    관리하는 건가요" = 0.4167).

런타임 비용
    이 파일을 미리 만들어 두면 매 턴 사용자 질문 1건만 임베딩하면 되므로 수십 ms 로 끝난다.
    (FAQ 임베딩을 요청 시점에 계산하면 236회 호출이 필요하다.)

실행:
    <intern_chatbot python> -X utf8 chatbot_demo_v2/scripts/build_faq_embeddings.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT.parent))

from chatbot_demo_v2.config.settings import load_settings          # noqa: E402
from chatbot_demo_v2.rag.adapter_util import prepare_ragcore_imports  # noqa: E402
from chatbot_demo_v2.scenario.matcher import normalize_text        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="FAQ 질문 임베딩 사전 계산")
    ap.add_argument("--out", default=str(PKG_ROOT / "data" / "faq_embeddings.json"))
    ap.add_argument("--batch-log", type=int, default=25)
    args = ap.parse_args()

    settings = load_settings()
    prepare_ragcore_imports(settings)
    from rag3.config import load_config
    from rag3.models import OllamaBackend

    config = load_config(str(settings.ragcore_config))
    backend = OllamaBackend(config)   # 임베딩은 항상 로컬(Gemini 백엔드도 embed 는 로컬 위임)

    faq = json.loads(settings.faq_path.read_text(encoding="utf-8"))
    entries = faq["entries"]
    print("FAQ %d행 임베딩 (%s)" % (len(entries), config.embedding_model))

    ids, texts = [], []
    for e in entries:
        q = e.get("question_normalized") or e.get("question") or ""
        if not q.strip():
            continue
        ids.append(e["id"])
        texts.append(q.strip())

    t0 = time.time()
    vecs: list[list[float]] = []
    for i, t in enumerate(texts):
        # 문서 프리픽스(embed_doc_prefix)로 색인 — 질의는 런타임에 is_query=True 로 임베딩된다.
        vecs.append(backend.embed([t], is_query=False)[0])
        if (i + 1) % args.batch_log == 0:
            print("  %d/%d (%.0fs)" % (i + 1, len(texts), time.time() - t0), flush=True)

    payload = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": config.embedding_model,
        "embed_doc_prefix": config.embed_doc_prefix,
        "dim": len(vecs[0]) if vecs else 0,
        "ids": ids,
        "questions": texts,
        "normalized": [normalize_text(t) for t in texts],
        "vectors": vecs,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("저장: %s  (%d건, dim=%d, %.1fs, %.1fMB)"
          % (out, len(ids), payload["dim"], time.time() - t0, out.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
