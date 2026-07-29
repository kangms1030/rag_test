"""파이프라인 다이어그램(SVG) 생성기.

`docs/` 에 3장을 만든다.
  - pipeline_v1.svg     : v1(chatbot_demo) 전진 DAG
  - pipeline_v2.svg     : v2 메인그래프 + RAG 서브그래프(사이클·HITL·메모리·스트리밍)
  - trace_timeline.svg  : LangSmith 공개 트레이스 2건의 실측 실행 타임라인

외부 의존성 없이 좌표를 직접 찍는다(mermaid-cli/graphviz 불필요).
    python -m chatbot_demo_v2.scripts.render_pipeline
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

DOCS = Path(__file__).resolve().parents[1] / "docs"

FONT = "'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif"

# (채움, 테두리) — 노드 종류별 색
KIND = {
    "det":   ("#e8f0fe", "#1d4ed8"),   # 결정론 노드(LLM 0회)
    "llm":   ("#fef3c7", "#b45309"),   # LLM 노드
    "rag":   ("#ede9fe", "#6d28d9"),   # RAG/서브그래프
    "hitl":  ("#fee2e2", "#b91c1c"),   # 사람 개입(interrupt)
    "dec":   ("#f1f5f9", "#475569"),   # 분기(조건부 엣지)
    "end":   ("#dcfce7", "#15803d"),   # 종단
    "note":  ("#ffffff", "#94a3b8"),   # 주석
    "black": ("#e5e7eb", "#374151"),   # 블랙박스
}
CYCLE = "#dc2626"      # 사이클 화살표
FLOW = "#475569"       # 일반 화살표
AUX = "#0f766e"        # 보조(메모리·스트리밍) 점선


class Canvas:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title = w, h, title
        self.parts: list[str] = []

    # --- 도형 ---------------------------------------------------------
    def box(self, x, y, w, h, title, sub="", kind="det", rx=10, fs=15):
        fill, stroke = KIND[kind]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
        )
        lines = [ln for ln in title.split("\n")]
        subs = [ln for ln in sub.split("\n") if ln] if sub else []
        total = len(lines) * (fs + 3) + len(subs) * 14
        cy = y + (h - total) / 2 + fs
        cx = x + w / 2
        for ln in lines:
            self.parts.append(
                f'<text x="{cx}" y="{cy:.1f}" font-family="{FONT}" font-size="{fs}" '
                f'font-weight="600" fill="#111827" text-anchor="middle">{escape(ln)}</text>'
            )
            cy += fs + 3
        for ln in subs:
            self.parts.append(
                f'<text x="{cx}" y="{cy:.1f}" font-family="{FONT}" font-size="11.5" '
                f'fill="#475569" text-anchor="middle">{escape(ln)}</text>'
            )
            cy += 14

    def group(self, x, y, w, h, label, color="#6d28d9", dash="6 4"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" '
            f'stroke="{color}" stroke-width="1.6" stroke-dasharray="{dash}"/>'
        )
        self.parts.append(
            f'<text x="{x + 14}" y="{y + 22}" font-family="{FONT}" font-size="14" '
            f'font-weight="700" fill="{color}">{escape(label)}</text>'
        )

    def pill(self, cx, cy, text, kind="end"):
        fill, stroke = KIND[kind]
        w = max(64, 12 * len(text) + 24)
        self.box(cx - w / 2, cy - 17, w, 34, text, kind=kind, rx=17, fs=14)

    # --- 화살표 -------------------------------------------------------
    def arrow(self, x1, y1, x2, y2, label="", color=FLOW, dashed=False, lx=None, ly=None):
        da = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.8"{da} marker-end="url(#a-{color[1:]})"/>'
        )
        if label:
            self.label(lx if lx is not None else (x1 + x2) / 2,
                       ly if ly is not None else (y1 + y2) / 2 - 5, label, color)

    def path(self, d, label="", color=CYCLE, dashed=False, lx=0, ly=0):
        da = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{da} '
            f'marker-end="url(#a-{color[1:]})"/>'
        )
        if label:
            self.label(lx, ly, label, color)

    def label(self, x, y, text, color="#334155", size=11.5, anchor="middle", weight="600"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'paint-order="stroke" stroke="#ffffff" stroke-width="3.5" '
            f'stroke-linejoin="round">{escape(text)}</text>'
        )

    def text(self, x, y, text, size=13, color="#111827", weight="400", anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    # --- 출력 ---------------------------------------------------------
    def save(self, path: Path):
        markers = "".join(
            f'<marker id="a-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
            for c in (FLOW, CYCLE, AUX)
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(self.title)}">'
            f'<defs>{markers}</defs>'
            f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
            + "".join(self.parts)
            + "</svg>"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")
        print(f"  wrote {path.relative_to(path.parents[2])}  ({len(svg) // 1024}KB)")


def legend(c: Canvas, x, y, items):
    for i, (kind, text) in enumerate(items):
        fill, stroke = KIND[kind]
        yy = y + i * 22
        c.parts.append(
            f'<rect x="{x}" y="{yy}" width="16" height="16" rx="4" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"/>'
        )
        c.text(x + 24, yy + 13, text, size=12, color="#334155")


# ======================================================================
# 1. v1 파이프라인
# ======================================================================
def render_v1():
    c = Canvas(880, 900, "chatbot_demo (v1) 파이프라인")
    c.text(40, 46, "v1 · chatbot_demo — 전진 DAG", size=22, weight="700")
    c.text(40, 72, "LangGraph 를 '관측 가능한 라우터'로만 사용: 사이클 0 · 인터럽트 0 · 대화메모리 0 · RAG 는 1노드 블랙박스",
           size=13, color="#6b7280")

    X, W = 320, 260
    c.pill(X + W / 2, 118, "START", "det")
    c.box(X, 150, W, 54, "normalize_input", "자유입력/버튼 정규화")
    c.box(X, 234, W, 54, "load_or_update_session", "세션·시나리오 상태 (턴 독립)")
    c.box(X, 318, W, 50, "select_input_kind", "버튼 / 자유입력", kind="dec")
    c.box(60, 318, 230, 50, "scenario_action_handler", "버튼 이동 (LLM 0회)")
    c.box(X, 402, W, 54, "scenario_matcher", "FAQ 유사도 매칭 (rapidfuzz)")
    c.box(X, 486, W, 50, "route_decider", "select_route: scenario / faq / rag", kind="dec")

    c.box(40, 576, 240, 56, "scenario_answer", "시나리오·FAQ 원문 그대로")
    c.box(320, 576, 240, 56, "rag3x_answer", "Rag3xEngine.ask()  ← 블랙박스", kind="black")
    c.box(600, 576, 240, 56, "web_search_answer", "(구조만, 기본 비활성)")

    c.box(X, 686, W, 54, "final_formatter", "응답 조립 (원문 그대로 노출)")
    c.pill(X + W / 2, 786, "END", "end")

    mid = X + W / 2
    for y1, y2 in ((135, 150), (204, 234), (288, 318)):
        c.arrow(mid, y1, mid, y2)
    c.arrow(X, 343, 290, 343, "버튼", lx=305, ly=336)
    c.arrow(mid, 368, mid, 402, "자유입력", lx=mid + 46, ly=388)
    c.arrow(175, 368, 175, 486, "", FLOW)   # 버튼 → 라우터
    c.arrow(mid, 456, mid, 486)
    c.arrow(mid, 536, 160, 576)
    c.arrow(mid, 536, mid, 576)
    c.arrow(mid, 536, 720, 576)
    c.arrow(160, 632, mid - 40, 686)
    c.arrow(mid, 632, mid, 686)
    c.arrow(720, 632, mid + 40, 686)
    c.arrow(mid, 740, mid, 769)

    c.group(40, 830, 800, 52, "", color="#94a3b8", dash="0")
    c.text(58, 862, "한계 — 답변은 DB 원문 낭독 · 진행상황 없음(단발 응답) · 이전 턴을 기억하지 않음 · "
                    "LangSmith 에 RAG 내부가 보이지 않음", size=12.5, color="#b91c1c", weight="600")
    c.save(DOCS / "pipeline_v1.svg")


# ======================================================================
# 2. v2 파이프라인
# ======================================================================
def render_v2():
    """v2 파이프라인.

    2026-07-28 수정 — 도식이 코드와 어긋나던 6곳을 바로잡았다.
      1. after_retrieve(검색 직후 3분기)를 빠뜨려 crag/no_answer 진입점이 하나뿐인 것처럼 보였다.
      2. 롤백 3종이 박스 하나로 뭉쳐 트리거 조건이 각각 다르다는 사실이 가려졌다.
      3. LangSmith 주석의 "9노드"가 grade_evidence 신설 후 갱신되지 않았다(실제 10노드).
      4. scenario_answer → final_formatter(시나리오 원문 직행) 엣지가 그려지지 않았다.
      5. scenario_action_handler → route_decider 화살표가 박스에 닿지 않고 허공에서 끊겼다.
      6. 에스컬레이션 사이클이 FAQ 경로 전용이라는 게이트가 표기되지 않았다.
    아울러 회색 박스 중 '노드가 아닌 것'(순수 라우팅 함수)을 명시했다.
    """
    c = Canvas(1820, 1400, "chatbot_demo_v2 파이프라인")
    c.text(40, 46, "v2 · chatbot_demo_v2 — 사이클 · HITL · 대화메모리 · 서브그래프", size=22, weight="700")
    c.text(40, 72, "LangGraph 를 제어구조로 사용. 결정론 경로(버튼·FAQ 정확일치)는 여전히 LLM 0회, "
                   "신규 LLM 노드는 전부 토글 + 실패 시 pass-through", size=13, color="#6b7280")

    # ---------- 메인그래프 ----------
    c.group(30, 96, 900, 1130, "메인그래프  ·  ChatState (+ messages)   —   노드 14개", color="#1d4ed8")

    MX, MW = 350, 250              # 중앙 컬럼
    mid = MX + MW / 2
    c.pill(mid, 140, "START", "det")
    c.box(MX, 172, MW, 52, "normalize_input", "정규화 + HumanMessage 적재")
    c.box(MX, 250, MW, 52, "load_or_update_session", "세션·시나리오 상태 (messages 는 보존)")
    c.box(MX, 328, MW, 46, "select_input_kind", "버튼 / 자유입력   · 라우팅 함수", kind="dec")
    c.box(70, 328, 230, 46, "scenario_action_handler", "버튼 이동 (LLM 0회)")
    c.box(MX, 400, MW, 54, "contextualize_query", "후속질문 → 독립질문 재작성", kind="llm")
    c.box(MX, 478, MW, 52, "scenario_matcher", "FAQ 의미 유사도 (점수·격차)")
    c.box(MX, 556, MW, 46, "route_decider", "scenario / faq / clarify / rag", kind="dec")

    c.box(60, 640, 230, 62, "clarify_node", "interrupt() — 후보 2개 되묻기\nCommand(resume) 로 재개", kind="hitl")
    c.box(340, 640, 230, 62, "scenario_answer", "시나리오·FAQ 원문 + 근거링크")
    c.box(620, 640, 250, 62, "rag3x_answer", "→ RAG 서브그래프 호출\nTTL 캐시 · 동시요청 락", kind="rag")

    c.box(620, 740, 250, 50, "rag_result_evaluator", "채택 / 웹검색 / 보류", kind="dec")
    c.box(620, 830, 250, 48, "web_search_answer", "(기본 비활성)")
    c.box(330, 830, 250, 56, "compose_answer", "근거 종합·상담체 재구성\n+ 숫자 대조 환각가드", kind="llm")
    c.box(330, 920, 250, 56, "answer_grader", "RESOLVED / UNRESOLVED\nFAQ→재시도 · RAG→신뢰도 강등", kind="llm")
    c.box(330, 1010, 250, 54, "final_formatter", "응답 조립 + AIMessage 적재")
    c.pill(455, 1110, "END", "end")

    # 엣지
    for y1, y2 in ((157, 172), (224, 250), (302, 328), (454, 478), (530, 556)):
        c.arrow(mid, y1, mid, y2)
    c.arrow(MX, 351, 300, 351, "버튼", lx=322, ly=344)
    c.arrow(mid, 374, mid, 400, "자유입력", lx=mid + 52, ly=392)
    # 수정 5 — 버튼 경로가 route_decider 왼쪽 변에 실제로 닿게 한다(기존엔 허공에서 끊겼다)
    c.path("M 185 374 L 185 579 L 345 579", "", color=FLOW)
    c.arrow(mid, 602, 175, 640, "애매·회색지대", lx=222, ly=622)
    c.arrow(mid, 602, mid + 20, 640, "시나리오/FAQ", lx=mid + 78, ly=632)
    c.arrow(mid, 602, 700, 640, "RAG", lx=650, ly=615)
    c.arrow(745, 702, 745, 740)
    c.arrow(745, 790, 745, 830, "웹검색", lx=790, ly=815)
    c.arrow(660, 790, 580, 838, "답변 있음", lx=612, ly=808)
    c.arrow(455, 702, 455, 830, "FAQ 합성", lx=505, ly=772)
    c.arrow(455, 886, 455, 920)
    c.arrow(455, 976, 455, 1010)
    c.arrow(455, 1064, 455, 1093)
    # 수정 4 — 시나리오 버튼 종단답변은 합성을 건너뛰고 곧바로 최종으로 간다(전용 엣지)
    c.path("M 340 690 L 314 690 L 314 1032 L 325 1032", "시나리오 원문 — 합성 안 함",
           color=FLOW, lx=205, ly=1004)
    # abstain → 최종
    c.path("M 870 765 L 900 765 L 900 1037 L 585 1037", "보류(abstain) → 최종",
           color=FLOW, lx=770, ly=1030)
    c.path("M 870 854 L 895 854 L 895 1030 L 585 1030", "", color=FLOW)
    # HITL 재개 분기
    c.arrow(290, 668, 340, 668, "후보 선택", lx=315, ly=628)
    c.path("M 175 702 L 175 762 L 610 762 L 700 740", "해당 없음 → RAG",
           color=FLOW, lx=372, ly=756)
    # 수정 6 — 에스컬레이션은 FAQ 경로에서만 발동한다.
    # 경로는 compose/grader 컬럼(~580)과 rag 컬럼(620~) 사이의 빈 통로(x=600)로 올린다
    # (기존 곡선은 web_search_answer 박스를 관통했다).
    c.path("M 580 948 L 600 948 L 600 712 L 696 704",
           "⟲ 에스컬레이션 (FAQ 경로만 · 미해결 · 예산 1회)", color=CYCLE, lx=712, ly=972)

    # 메모리 패널
    c.box(60, 1120, 250, 76, "InMemorySaver", "thread_id = 세션:epoch\nmessages(add_messages) 누적\n새로고침/reset → 새 epoch",
          kind="note")
    c.arrow(185, 1120, 185, 1090, "", AUX, dashed=True)
    c.path("M 310 1150 L 620 1150 L 620 1064 L 585 1050", "", color=AUX, dashed=True)
    c.label(430, 1168, "체크포인터가 턴 사이 대화를 유지", AUX)

    # ---------- RAG 서브그래프 ----------
    c.group(960, 96, 830, 1130, "RAG 서브그래프  ·  rag_subgraph / RagState   —   노드 10개",
            color="#6d28d9")
    c.text(978, 42, "controller_x 의 S0~S8 을 10개 노드로 재편성 "
                    "(★ grade_evidence 는 2026-07-27 신설). 회색 중 after_* 는 노드가 아닌 라우팅 함수.",
           size=12.5, color="#6b7280")

    SX, SW = 1250, 250
    smid = SX + SW / 2             # 1375
    c.box(SX, 128, SW, 42, "prepare", "질문·예산·deadline 초기화")
    c.box(SX, 190, SW, 48, "retrieve", "임베딩 → 청크검색 → 리랭킹 (top 6)")
    # 수정 1 — 검색 직후에도 3분기가 있다(도식에서 빠져 있었다)
    c.box(SX, 258, SW, 44, "after_retrieve", "grade / crag / no_answer  · 라우팅 함수", kind="dec")
    # 2026-07-27 신설 — 검색 랭킹만으로 못 고치는 실패를 의미 판정으로 잡는다
    c.box(SX, 322, SW, 58, "grade_evidence ★",
          "근거 3등급 판정 (LLM 1회)\nprimary / supporting / irrelevant", kind="llm")
    c.box(SX, 400, SW, 44, "after_grade", "answer / crag / no_answer  · 라우팅 함수", kind="dec")
    c.box(SX, 464, SW, 58, "answer_node",
          "등급별 예산 배분 후 답변\nprimary 전문 · supporting 축약")
    c.box(SX, 542, SW, 48, "verify_node", "근거 정합성 · 숫자 대조")
    c.box(SX, 610, SW, 44, "after_verify", "rollback A/B/C / done  · 라우팅 함수", kind="dec")
    c.box(1005, 300, 205, 62, "crag_rewrite", "질의 재작성 (judge)\n결과가 더 나쁘면 원본 유지")
    # 수정 2 — 롤백 3종은 트리거가 각각 다른 독립 노드다
    c.box(1005, 690, 240, 78, "rollback_top1", "A · 회피 ∧ top ≥ 0.5\n등급 통과 1순위로 재생성", kind="rag")
    c.box(1255, 690, 240, 78, "rollback_vision", "B · 숫자 미지원 ∧ 스캔본\ntext → vision 교차확인", kind="rag")
    c.box(1505, 690, 240, 78, "rollback_ocr", "C · 전사 ≠ OCR\nvision → 텍스트 재구성", kind="rag")
    c.box(SX, 830, SW, 48, "finalize", "근거 해석 · 이미지 사본 · 정규화")
    c.pill(smid, 916, "END", "end")

    # 스파인
    c.arrow(smid, 170, smid, 190)
    c.arrow(smid, 238, smid, 258)
    c.arrow(smid, 302, smid, 322, "근거 있음", lx=smid + 62, ly=316)
    c.arrow(smid, 380, smid, 400)
    c.arrow(smid, 444, smid, 464, "근거 남음", lx=smid + 62, ly=458)
    c.arrow(smid, 522, smid, 542)
    c.arrow(smid, 590, smid, 610)
    # CRAG 진입 2곳 (검색 직후 · 등급 직후)
    c.arrow(1248, 294, 1214, 302, "경계 점수", lx=1120, ly=288)
    c.arrow(1248, 408, 1162, 366, "전부 무관", lx=1198, ly=392)
    c.path("M 1107 300 C 1052 250 1062 198 1244 212", "⟲ CRAG 사이클 (1회)", color=CYCLE,
           lx=1062, ly=196)
    # 보류(no_answer) — 왼쪽 통로로 finalize 직행. 두 판정 지점이 같은 통로를 쓴다.
    c.path("M 1250 268 L 985 268 L 985 854 L 1245 854", "근거 없음 → 보류", color=FLOW,
           lx=1062, ly=560)
    c.parts.append(f'<line x1="1250" y1="438" x2="985" y2="438" stroke="{FLOW}" stroke-width="2"/>')
    # 롤백 3종 진입
    c.arrow(1290, 654, 1160, 688, "A", lx=1198, ly=676)
    c.arrow(smid, 654, smid, 688, "B", lx=smid + 20, ly=676)
    c.arrow(1460, 654, 1600, 688, "C", lx=1556, ly=676)
    # 정상 종료(done) — 오른쪽 통로
    c.path("M 1500 632 L 1768 632 L 1768 854 L 1505 854", "정상 종료", color=FLOW,
           lx=1632, ly=624)
    # 롤백 → finalize (셋 다 단발, 되돌아가지 않는다)
    c.path("M 1125 768 L 1125 800 L 1300 800 L 1300 828", "", color=FLOW)
    c.arrow(smid, 768, smid, 828)
    c.path("M 1625 768 L 1625 800 L 1450 800 L 1450 828", "", color=FLOW)
    c.arrow(smid, 878, smid, 899)

    # 스트리밍 패널
    c.box(980, 1000, 790, 62, "get_stream_writer() → SSE progress",
          "retrieve / grade / crag / answer / verify / rollback / compose 단계가 브라우저에 실시간 표시",
          kind="note")
    # 수정 3 — grade_evidence 신설 반영(9 → 10)
    c.box(980, 1080, 790, 76, "LangSmith child run",
          "v1 은 rag3x.ask 1개만 보였지만 v2 는 위 10개 노드가 개별 run 으로 기록된다\n"
          "(실측: chat_turn → rag3x_answer → rag3x.ask → rag_subgraph → prepare/retrieve/…)",
          kind="note")

    # 범례
    legend(c, 60, 1248, [
        ("det", "결정론 노드 — LLM 0회"),
        ("llm", "LLM 노드 — .env 토글, 실패 시 pass-through"),
        ("rag", "RAG 서브그래프"),
        ("hitl", "HITL — interrupt() 로 사람에게 되묻기"),
    ])
    legend(c, 620, 1248, [
        ("dec", "분기 판정 — ‘라우팅 함수’ 표기는 노드가 아님(run·SSE 미기록)"),
        ("note", "부가 설명 — 제어흐름 아님"),
    ])
    c.parts.append(f'<line x1="620" y1="1302" x2="680" y2="1302" stroke="{CYCLE}" stroke-width="2"/>')
    c.text(690, 1307, "사이클 (예산으로 상한)", size=12, color="#334155")
    c.parts.append(f'<line x1="620" y1="1324" x2="680" y2="1324" stroke="{AUX}" stroke-width="2" '
                   f'stroke-dasharray="5 4"/>')
    c.text(690, 1329, "체크포인터 메모리 (제어흐름 아님)", size=12, color="#334155")
    c.text(60, 1372, "사이클 예산 — CRAG 1회 · 롤백 A/B/C 각 1회 · 에스컬레이션 1회(FAQ 경로만). "
                     "여기에 rag3 원본의 모델호출 상한(<5)·deadline(240초) 이 그대로 적용된다. "
                     "되돌아가는 엣지는 CRAG·에스컬레이션 둘뿐이고, 롤백은 단발 후 finalize 로 빠진다.",
           size=12.5, color="#6b7280")
    c.save(DOCS / "pipeline_v2.svg")


# ======================================================================
# 3. 실측 트레이스 타임라인
# ======================================================================
# LangSmith 공개 트레이스에서 추출한 실측치(초).
TRACE1 = [
    ("normalize_input", 0.00, 0, "det"),
    ("load_or_update_session", 0.00, 0, "det"),
    ("contextualize_query", 0.00, 0, "llm"),
    ("scenario_matcher", 0.00, 0, "det"),
    ("route_decider", 0.00, 0, "det"),
    ("rag3x_answer  (엔진 콜드 초기화 ≈31s 포함)", 38.25, 0, "rag"),
    ("└ rag_subgraph · prepare", 0.00, 1, "rag"),
    ("└ rag_subgraph · retrieve  (top 0.054 < floor 0.1)", 4.24, 1, "rag"),
    ("└ rag_subgraph · crag_rewrite  ⟲ 사이클 발동", 1.07, 1, "cycle"),
    ("└ rag_subgraph · retrieve (2회차, top 0.063)", 1.76, 1, "rag"),
    ("└ rag_subgraph · finalize  → answer_path=none", 0.00, 1, "rag"),
    ("rag_result_evaluator → 보류(abstain)", 0.00, 0, "dec"),
    ("final_formatter", 0.00, 0, "det"),
]
TRACE2 = [
    ("normalize_input", 0.00, 0, "det"),
    ("load_or_update_session", 0.00, 0, "det"),
    ("contextualize_query  ← 이전 턴 기억 반영", 1.04, 0, "llm"),
    ("scenario_matcher", 0.00, 0, "det"),
    ("route_decider", 0.00, 0, "det"),
    ("rag3x_answer  (엔진 예열됨)", 68.36, 0, "rag"),
    ("└ rag_subgraph · prepare", 0.00, 1, "rag"),
    ("└ rag_subgraph · retrieve  (top 0.114 ≥ floor)", 2.96, 1, "rag"),
    ("└ rag_subgraph · answer_node  (text 경로)", 1.19, 1, "rag"),
    ("└ rag_subgraph · verify_node  ★ 병목", 64.16, 1, "slow"),
    ("└ rag_subgraph · finalize  → confidence=high", 0.00, 1, "rag"),
    ("rag_result_evaluator → 채택", 0.00, 0, "dec"),
    ("compose_answer  → composed=true (891자)", 5.03, 0, "llm"),
    ("answer_grader  (당시엔 RAG 경로 생략 — 현재는 판정함)", 0.00, 0, "dec"),
    ("final_formatter", 0.00, 0, "det"),
]
BAR = {"det": "#3b82f6", "llm": "#f59e0b", "rag": "#8b5cf6",
       "dec": "#64748b", "cycle": "#dc2626", "slow": "#e11d48"}


def render_trace():
    rows = len(TRACE1) + len(TRACE2)
    c = Canvas(1360, 200 + rows * 30 + 190, "LangSmith 실측 트레이스")
    c.text(40, 46, "실측 LangSmith 트레이스 — 같은 세션의 연속 2턴", size=22, weight="700")
    c.text(40, 72, "프로젝트 school-network-chatbot-demo-v2 · 2026-07-24 · 공개 공유 링크에서 추출",
           size=13, color="#6b7280")

    x0, scale = 640, 8.0     # 1초 = 8px
    y = 118

    def block(title, sub, data, y):
        c.box(40, y, 1280, 46, title, sub, kind="note", rx=8, fs=15)
        y += 62
        for name, dur, depth, kind in data:
            c.text(56 + depth * 16, y + 12, name, size=12.5,
                   color="#111827" if depth == 0 else "#475569",
                   weight="600" if depth == 0 else "400")
            w = max(3.0, dur * scale)
            c.parts.append(
                f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="16" rx="4" '
                f'fill="{BAR[kind]}" opacity="0.9"/>'
            )
            c.text(x0 + w + 8, y + 13, f"{dur:.2f}s", size=11.5, color="#6b7280")
            y += 30
        return y + 16

    y = block("턴 1 — “아이폰에 비해 갤럭시 기기들만 인터넷 속도가 느려. 왜 그런거고 해결 방법을 알려줄래?”",
              "총 38.52s · route=abstain · CRAG 사이클 1회 발동 후 근거 부족으로 보류 (리랭크 0.063 < floor 0.1)",
              TRACE1, y)
    y = block("턴 2 — “나는 대구에 있는 고등학교에 다니는 선생님이야”  (같은 thread_id)",
              "총 74.48s · contextualize 가 이력을 반영해 “…갤럭시 기기의 인터넷 속도가 느린 문제에 대해 어디로 문의해야 하나요?”"
              " 로 재작성 → RAG 채택(high) → 합성",
              TRACE2, y)

    # 눈금
    c.parts.append(f'<line x1="{x0}" y1="106" x2="{x0}" y2="{y - 16}" stroke="#e5e7eb" stroke-width="1"/>')
    for s in range(0, 80, 10):
        gx = x0 + s * scale
        c.parts.append(f'<line x1="{gx}" y1="106" x2="{gx}" y2="{y - 16}" stroke="#f1f5f9" stroke-width="1"/>')
        c.text(gx, 100, f"{s}s", size=11, color="#94a3b8", anchor="middle")

    c.box(40, y, 1280, 118, "이 트레이스가 증명하는 것",
          "① LangGraph 사이클이 실제로 발동한다 — 턴1에서 retrieve → crag_rewrite → retrieve 로 재검색 후 예산 소진(1→0)으로 정지\n"
          "② 대화 메모리가 작동한다 — 턴2의 “나는 …선생님이야” 한 마디가 이전 턴 주제와 합쳐져 독립 질문으로 재작성됨\n"
          "③ RAG 내부가 관측된다 — v1 에서 1개 블랙박스였던 구간이 prepare/retrieve/crag/answer/verify/finalize 로 분해되어 기록됨\n"
          "④ 환각가드가 통과했다 — verify unsupported_claims=[] · composer 합성 채택(composed=true, fallback 없음)\n"
          "⑤ 병목이 드러난다 — 턴2의 64.16s 는 verify_node 단일 Gemini 호출(백오프 추정). 턴1의 38s 중 31s 는 엔진 콜드 초기화\n"
          "⑥ 주의 — 이 트레이스는 2026-07-24 기록이라 grade_evidence(07-27 신설) 이전이다. 현재 서브그래프는 10노드다",
          kind="note", rx=10, fs=15)
    c.save(DOCS / "trace_timeline.svg")


def main() -> None:
    print("파이프라인 다이어그램 생성:")
    render_v1()
    render_v2()
    render_trace()


if __name__ == "__main__":
    main()
