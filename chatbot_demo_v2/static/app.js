"use strict";

// ---- 세션 ----
let sessionId = sessionStorage.getItem("chatbot_demo_v2_sid") || null;
function setSession(sid) {
  if (sid) { sessionId = sid; sessionStorage.setItem("chatbot_demo_v2_sid", sid); }
}

const $ = (id) => document.getElementById(id);
const chatEl = $("chat");
let busyTimer = null;
let inFlight = false;
let activeMsg = null;

// ---- 유틸 ----
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function addUser(text) {
  const m = el("div", "msg user", text);
  chatEl.appendChild(m);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ---- 안전한 markdown-lite 렌더 ----
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
// 답변에 실린 출처 표기 [p53] → 해당 근거 이미지로 가는 칩. 2026-07-27(작업 9).
// 서버가 근거 페이지와 대조해 유효한 표기만 남기므로 여기서는 렌더링만 한다.
let evidenceByPage = {};   // page_number -> image_url
function setEvidenceIndex(resp) {
  evidenceByPage = {};
  [...(resp.evidence || []), ...(resp.faq_evidence || [])].forEach((e) => {
    if (e && e.page_number != null && e.image_url) evidenceByPage[e.page_number] = e.image_url;
  });
}
function inlineFmt(s) {
  let html = escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[p(\d{1,4})\]/g, (m, pg) => {
    const url = evidenceByPage[Number(pg)];
    if (!url) return "";                       // 근거 이미지가 없으면 표기를 감춘다
    return '<button class="cite" data-img="' + escapeHtml(url) + '" title="근거 보기">p'
      + escapeHtml(pg) + "</button>";
  });
  return html;
}
function renderAnswer(container, text) {
  const lines = String(text || "").split("\n");
  let ul = null;
  const flushUl = () => { if (ul) { container.appendChild(ul); ul = null; } };
  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) { flushUl(); return; }
    const bullet = line.match(/^[*\-]\s+(.*)$/);
    if (bullet) {
      if (!ul) ul = el("ul");
      const li = el("li"); li.innerHTML = inlineFmt(bullet[1]); ul.appendChild(li);
      return;
    }
    flushUl();
    const headOnly = line.match(/^\*\*(.+?)\*\*[:：]?$/);
    if (headOnly) {
      const h = el("div", "ans-h"); h.textContent = headOnly[1]; container.appendChild(h);
      return;
    }
    const p = el("p"); p.innerHTML = inlineFmt(line); container.appendChild(p);
  });
  flushUl();
}

function confBadgeClass(c) {
  if (c === "high") return "badge high";
  if (c === "low" || c === "unknown") return "badge low";
  if (c === "abstain" || c === "none") return "badge abstain";
  return "badge";
}
function confLabel(c) {
  return ({ high: "높음", low: "낮음", unknown: "불명", abstain: "회피", none: "없음" }[c]) || c;
}

// ---- 봇 답변 말풍선 ----
function addBot(resp) {
  const wrap = el("div", "msg bot");
  const ansBox = el("div", "answer");
  setEvidenceIndex(resp);                       // [p53] 칩이 참조할 근거 인덱스
  renderAnswer(ansBox, resp.answer || "(응답 없음)");
  ansBox.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".cite");
    if (btn) openLightbox(btn.dataset.img);
  });
  wrap.appendChild(ansBox);

  // 합성된 답변이면 원문(저장된 모범답변) 접기로 함께 제공
  if (resp.composed && resp.original_answer) {
    const det = el("details", "orig");
    det.appendChild(el("summary", null, "원문 보기 (저장된 모범답변)"));
    const ob = el("div", "orig-body");
    renderAnswer(ob, resp.original_answer);
    det.appendChild(ob);
    wrap.appendChild(det);
  }

  // route / 신뢰도 한 줄 + 피드백
  const foot = el("div", "msg-foot");
  const r = el("div", "msg-route");
  if (resp.route) {
    r.innerHTML = '<span class="dot">●</span> ' + escapeHtml(resp.route)
      + (resp.confidence ? " · 신뢰도 " + escapeHtml(confLabel(resp.confidence)) : "")
      + (resp.composed ? " · <span class='mini-tag'>정리됨</span>" : "");
  }
  foot.appendChild(r);
  if (resp.run_id) foot.appendChild(feedbackBox(resp.run_id));
  wrap.appendChild(foot);

  wrap._resp = resp;
  wrap.addEventListener("click", () => selectMessage(wrap));
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;

  selectMessage(wrap);
  renderScenarioOptions(resp.options, resp.scenario);
}

// ---- 👍 / 👎 피드백 ----
function feedbackBox(runId) {
  const box = el("div", "fb");
  const mk = (label, score, title) => {
    const b = el("button", "fb-btn", label);
    b.title = title;
    b.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      box.querySelectorAll(".fb-btn").forEach((x) => (x.disabled = true));
      try {
        const res = await post("/api/feedback", { run_id: runId, score });
        box.appendChild(el("span", "fb-msg",
          res.recorded ? " 기록됨" : " (LangSmith 비활성 — 미기록)"));
      } catch (e) {
        box.appendChild(el("span", "fb-msg", " 전송 실패"));
      }
    });
    return b;
  };
  box.appendChild(mk("👍", 1, "도움이 됐어요"));
  box.appendChild(mk("👎", 0, "도움이 안 됐어요"));
  return box;
}

// ---- clarify 되묻기 ----
function addClarify(payload) {
  const wrap = el("div", "msg bot clarify");
  wrap.appendChild(el("div", "clarify-title", "어떤 상황인지 확인이 필요해요"));
  wrap.appendChild(el("div", "clarify-sub",
    "비슷한 문의가 여러 건 있어요. 해당하는 항목을 골라 주세요."));
  const box = el("div", "clarify-opts");
  (payload.candidates || []).forEach((cd) => {
    const b = el("button", "clarify-btn");
    b.appendChild(el("div", "cb-q", cd.question));
    b.appendChild(el("div", "cb-score", "유사도 " + Number(cd.score ?? 0).toFixed(2)));
    b.addEventListener("click", () => {
      box.querySelectorAll("button").forEach((x) => (x.disabled = true));
      sendClarify(cd.faq_id, cd.question);
    });
    box.appendChild(b);
  });
  const none = el("button", "clarify-btn none", "해당 없음 — 자료를 직접 검색해 주세요");
  none.addEventListener("click", () => {
    box.querySelectorAll("button").forEach((x) => (x.disabled = true));
    sendClarify("__none__", "해당 없음");
  });
  box.appendChild(none);
  wrap.appendChild(box);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function selectMessage(wrap) {
  if (activeMsg) activeMsg.classList.remove("active");
  activeMsg = wrap;
  wrap.classList.add("active");
  renderInspector(wrap._resp);
}

// ---- 우측 인스펙터 ----
function renderInspector(resp) {
  if (!resp) return;
  $("insp-empty").classList.add("hidden");

  const badges = $("insp-badges");
  badges.innerHTML = "";
  if (resp.route) badges.appendChild(el("span", "badge route", "route: " + resp.route));
  if (resp.answer_source) badges.appendChild(el("span", "badge", "출처: " + resp.answer_source));
  if (resp.answer_path) badges.appendChild(el("span", "badge", "path: " + resp.answer_path));
  if (resp.confidence) badges.appendChild(el("span", confBadgeClass(resp.confidence), "신뢰도: " + confLabel(resp.confidence)));
  if (resp.composed) badges.appendChild(el("span", "badge ok", "답변 정리됨"));
  if (resp.grader_verdict) badges.appendChild(el("span", "badge", "판정: " + resp.grader_verdict));

  // 파이프라인
  const secPipe = $("sec-pipeline");
  const pipe = $("pipeline");
  pipe.innerHTML = "";
  const trace = resp.trace || [];
  if (trace.length) {
    trace.forEach((t) => {
      const li = el("li");
      li.appendChild(el("div", "p-node", t.node));
      if (t.detail) li.appendChild(el("div", "p-detail", t.detail));
      pipe.appendChild(li);
    });
    secPipe.classList.remove("hidden");
  } else {
    secPipe.classList.add("hidden");
  }

  // 메타
  const kv = $("insp-kv");
  kv.innerHTML = "";
  const addKV = (k, v) => {
    if (v === undefined || v === null || v === "") return;
    kv.appendChild(el("div", "k", k));
    kv.appendChild(el("div", "v", String(v)));
  };
  addKV("소요 시간(초)", resp.elapsed_seconds);
  const tm = resp.timings || {};
  if (tm.rag_s) addKV("RAG 시간(초)", Number(tm.rag_s).toFixed(1));
  if (tm.compose_s) addKV("답변 정리(초)", Number(tm.compose_s).toFixed(1));
  if (resp.composer_fallback) addKV("합성 폐기 사유", resp.composer_fallback);
  const sm = resp.source_meta || {};
  if (sm.type === "faq") {
    addKV("엑셀 시트", sm.sheet);
    addKV("행", sm.row);
    addKV("질문 유형", sm.question_type);
    addKV("장애 유형", sm.fault_type);
    addKV("유사도", sm.best_score !== undefined ? Number(sm.best_score).toFixed(3) : null);
    if (sm.source_files && sm.source_files.length) addKV("인용 표기", sm.source_files.join(", "));
    // 코퍼스에서 실제 식별된 근거 문서(쪽번호 미인용이면 문서명만)
    if (sm.evidence_docs && sm.evidence_docs.length) {
      addKV("근거 문서", sm.evidence_docs.map((d) =>
        d.document_name + (d.pages && d.pages.length ? " p" + d.pages.join(",") : " (쪽 미인용)")
      ).join(" · "));
    }
  }
  if (sm.type === "rag3x") {
    addKV("리랭크 점수", sm.rerank_top_score !== undefined && sm.rerank_top_score !== null ? Number(sm.rerank_top_score).toFixed(4) : null);
    addKV("RAG route", sm.route_reason);
    if (sm.metrics && sm.metrics.timings_seconds) {
      addKV("검색 시간(초)", sm.metrics.timings_seconds.retrieve);
      addKV("생성 시간(초)", sm.metrics.timings_seconds.answer);
    }
  }
  if (sm.type === "web") {
    addKV("provider", sm.provider);
    addKV("모델", sm.model);
    if (sm.search_queries && sm.search_queries.length) addKV("검색어", sm.search_queries.join(" · "));
    if (sm.usage) addKV("검색 횟수", sm.usage.searches);
    if (sm.sources && sm.sources.length) {
      kv.appendChild(el("div", "k", "출처"));
      const box = el("div", "v");
      sm.sources.forEach((s) => {
        const a = el("a", null, s.title || s.url);
        a.href = s.url; a.target = "_blank"; a.rel = "noopener noreferrer";
        box.appendChild(a);
        box.appendChild(el("div"));
      });
      kv.appendChild(box);
    }
  }
  // Google 검색 grounding 이용약관: 응답과 함께 '검색 추천'을 표시해야 한다.
  const sugg = $("insp-search-suggest");
  if (sm.type === "web" && sm.search_entry_point) {
    sugg.innerHTML = sm.search_entry_point;
    sugg.classList.remove("hidden");
  } else {
    sugg.innerHTML = "";
    sugg.classList.add("hidden");
  }
  $("sec-meta").classList.toggle("hidden", kv.children.length === 0 && sugg.classList.contains("hidden"));

  // 근거 이미지 (RAG evidence + FAQ 근거 페이지)
  const eviBox = $("insp-evi");
  eviBox.innerHTML = "";
  const evis = [
    ...(resp.evidence || []).filter((e) => e.image_url),
    ...(resp.faq_evidence || []).filter((e) => e.image_url),
  ];
  if (evis.length) {
    evis.forEach((e) => {
      const box = el("div");
      const img = el("img");
      img.src = e.image_url; img.loading = "lazy";
      img.addEventListener("click", () => openLightbox(e.image_url));
      box.appendChild(img);
      box.appendChild(el("div", "evi-cap", (e.document_name || "") + " p" + (e.page_number ?? "?")));
      eviBox.appendChild(box);
    });
    $("sec-evi").classList.remove("hidden");
  } else {
    $("sec-evi").classList.add("hidden");
  }

  // 검증 · 경고
  const flags = $("insp-flags");
  flags.innerHTML = "";
  let hasFlags = false;
  if (resp.verification) {
    const v = resp.verification;
    const f = [];
    if (v.abstain) f.push("회피");
    if (v.transcription_ocr_mismatch && v.transcription_ocr_mismatch.length) f.push("전사-OCR 불일치");
    flags.appendChild(el("div", "flag-line", "검증: " + (f.length ? f.join(", ") : "이상 없음")));
    hasFlags = true;
  }
  (resp.warnings || []).forEach((w) => { flags.appendChild(el("div", "warn-box", "⚠ " + w)); hasFlags = true; });
  $("sec-flags").classList.toggle("hidden", !hasFlags);
}

// ---- 시나리오 칩 ----
function renderScenarioOptions(options, scenario) {
  const box = $("scenario-options");
  box.innerHTML = "";
  const prompt = $("scenario-prompt");
  if (scenario && scenario.node_id) {
    prompt.textContent = "현재: " + (scenario.scenario_id || "") + " / " + scenario.node_id
      + (scenario.completed ? " (완료)" : "");
  } else {
    prompt.textContent = "";
  }
  (options || []).forEach((o) => {
    const isRestart = o.option_id === "__restart__";
    const b = el("button", "chip" + (isRestart ? " restart" : ""), o.label);
    b.addEventListener("click", () => sendAction(o));
    box.appendChild(b);
  });
}

// ---- 진행 표시 ----
function setBusy(on, label) {
  inFlight = on;
  $("busy").classList.toggle("hidden", !on);
  $("btn-send").disabled = on;
  if (on) {
    $("busy-text").textContent = label || "처리 중…";
    $("busy-steps").innerHTML = "";
    const t0 = Date.now();
    busyTimer = setInterval(() => {
      $("busy-timer").textContent = ((Date.now() - t0) / 1000).toFixed(1) + "s";
    }, 200);
  } else if (busyTimer) {
    clearInterval(busyTimer); busyTimer = null; $("busy-timer").textContent = "";
  }
}

function pushStep(msg) {
  $("busy-text").textContent = msg;
  const s = el("div", "step", msg);
  const box = $("busy-steps");
  box.appendChild(s);
  while (box.children.length > 4) box.removeChild(box.firstChild);
}

// ---- 통신 ----
async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.detail || ("HTTP " + res.status));
    err.status = res.status;
    throw err;
  }
  return data;
}

/** SSE 스트리밍 대화. progress/node 는 진행표시, final/clarify/error 로 종료. */
async function streamChat(body) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = "HTTP " + res.status;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    const err = new Error(detail); err.status = res.status; throw err;
  }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const blocks = buf.split("\n\n");
    buf = blocks.pop();                       // 미완성 블록은 남겨둠
    for (const block of blocks) {
      if (!block.trim()) continue;
      let ev = null, data = null;
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) ev = line.slice(7).trim();
        else if (line.startsWith("data: ")) {
          try { data = JSON.parse(line.slice(6)); } catch (e) { data = null; }
        }
      }
      if (!ev) continue;
      if (ev === "progress") { if (data && data.msg) pushStep(data.msg); }
      else if (ev === "node") { /* 필요 시 노드 진행 시각화 */ }
      else if (ev === "final") { setSession(data.session_id); addBot(data); return; }
      else if (ev === "clarify") { setSession(data.session_id); addClarify(data); return; }
      else if (ev === "error") {
        const err = new Error(data.detail || "오류"); err.status = data.status; throw err;
      }
    }
  }
}

async function sendMessage(text) {
  if (inFlight) return;
  addUser(text);
  setBusy(true, "질문을 확인하고 있어요…");
  try {
    await streamChat({ session_id: sessionId, message: text });
  } catch (e) { handleError(e); } finally { setBusy(false); }
}

async function sendAction(opt) {
  if (inFlight) return;
  addUser("▶ " + opt.label);
  setBusy(true, "시나리오 이동 중…");
  try {
    await streamChat({
      session_id: sessionId,
      action: {
        type: "scenario_option",
        scenario_id: opt.scenario_id,
        node_id: opt.node_id,
        option_id: opt.option_id,
        label: opt.label,
      },
    });
  } catch (e) { handleError(e); } finally { setBusy(false); }
}

async function sendClarify(choice, label) {
  if (inFlight) return;
  addUser("▶ " + label);
  setBusy(true, "선택하신 내용으로 답변을 준비하고 있어요…");
  try {
    await streamChat({ session_id: sessionId, clarify_response: { choice } });
  } catch (e) { handleError(e); } finally { setBusy(false); }
}

function handleError(e) {
  const msg = e.status === 429
    ? "이미 다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요."
    : e.status === 503
    ? "RAG 엔진을 사용할 수 없습니다. 오른쪽에서 엔진 예열을 하거나 관리자에게 문의하세요."
    : (e.message || "오류가 발생했습니다.");
  const b = el("div", "msg bot");
  b.appendChild(el("div", "error-banner", "⚠ " + msg));
  chatEl.appendChild(b);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ---- 라이트박스 ----
function openLightbox(src) {
  let lb = $("lightbox");
  if (!lb) {
    lb = el("div", "lightbox hidden"); lb.id = "lightbox";
    lb.addEventListener("click", () => lb.classList.add("hidden"));
    const img = el("img"); img.id = "lightbox-img";
    lb.appendChild(img); document.body.appendChild(lb);
  }
  $("lightbox-img").src = src;
  lb.classList.remove("hidden");
}

// ---- 초기 로드 ----
async function loadRoot() {
  try {
    const root = await (await fetch("/api/scenarios/root")).json();
    renderScenarioOptions(root.options, { scenario_id: root.scenario_id, node_id: root.node_id });
  } catch (e) {
    $("scenario-options").textContent = "시나리오 로드 실패";
  }
}

async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("status").textContent = "● 준비 완료";
    const eng = h.engine || {};
    const ls = h.langsmith || {};
    const tg = h.toggles || {};
    const on = (b) => (b ? "on" : "off");
    $("engine-status").innerHTML =
      "RAG 엔진: <b>" + (eng.status || "?") + "</b>" +
      (eng.error ? " (" + eng.error + ")" : "") +
      " · 백엔드: " + (h.routing && h.routing.backend) +
      " · LangSmith: " + on(ls.tracing_enabled) +
      "<br>정리(composer): " + on(tg.composer_faq || tg.composer_rag) +
      " · 되묻기(clarify): " + on(tg.clarify) +
      " · 판정(grader): " + on(tg.grader) +
      " · 웹검색: " + on(h.web_search && h.web_search.enabled);
  } catch (e) {
    $("status").textContent = "● 서버 연결 실패";
  }
}

async function warmup() {
  $("warmup-msg").textContent = "예열 시작…";
  try {
    const r = await post("/api/warmup", { deep: true });
    $("warmup-msg").textContent = "예열 요청됨 (상태: " + r.status + ")";
    setTimeout(loadHealth, 3000);
  } catch (e) {
    $("warmup-msg").textContent = "예열 실패: " + (e.message || "");
  }
}

async function resetSession() {
  if (sessionId) {
    try { await post("/api/reset", { session_id: sessionId }); } catch (e) {}
  }
  chatEl.innerHTML = "";
  activeMsg = null;
  $("insp-badges").innerHTML = "";
  ["sec-pipeline", "sec-meta", "sec-evi", "sec-flags"].forEach((s) => $(s).classList.add("hidden"));
  $("insp-empty").classList.remove("hidden");
  loadRoot();
}

// ---- 이벤트 바인딩 ----
$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const inp = $("chat-input");
  const text = inp.value.trim();
  if (!text) return;
  inp.value = "";
  sendMessage(text);
});
$("btn-home").addEventListener("click", resetSession);
$("btn-warmup").addEventListener("click", warmup);

loadHealth();
loadRoot();
