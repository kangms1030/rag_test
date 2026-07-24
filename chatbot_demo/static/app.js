"use strict";

// ---- 세션 ----
let sessionId = sessionStorage.getItem("chatbot_demo_sid") || null;
function setSession(sid) {
  if (sid) { sessionId = sid; sessionStorage.setItem("chatbot_demo_sid", sid); }
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

// ---- 안전한 markdown-lite 렌더 (RAG 답변의 **굵게** / * 목록 처리) ----
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
function inlineFmt(s) {
  return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
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

// ---- 좌측: 봇 답변 말풍선 (답변 텍스트 + route 한 줄) ----
function addBot(resp) {
  const wrap = el("div", "msg bot");
  const ansBox = el("div", "answer");
  renderAnswer(ansBox, resp.answer || "(응답 없음)");
  wrap.appendChild(ansBox);

  if (resp.route) {
    const r = el("div", "msg-route");
    r.innerHTML = '<span class="dot">●</span> ' + escapeHtml(resp.route)
      + (resp.confidence ? " · 신뢰도 " + escapeHtml(confLabel(resp.confidence)) : "");
    wrap.appendChild(r);
  }

  // 말풍선 클릭 → 우측 인스펙터에 해당 답변의 근거/과정 표시
  wrap._resp = resp;
  wrap.addEventListener("click", () => selectMessage(wrap));
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;

  selectMessage(wrap);
  renderScenarioOptions(resp.options, resp.scenario);
}

function selectMessage(wrap) {
  if (activeMsg) activeMsg.classList.remove("active");
  activeMsg = wrap;
  wrap.classList.add("active");
  renderInspector(wrap._resp);
}

// ---- 우측: 인스펙터 (근거 · 처리 과정) ----
function renderInspector(resp) {
  if (!resp) return;
  $("insp-empty").classList.add("hidden");

  // 배지
  const badges = $("insp-badges");
  badges.innerHTML = "";
  if (resp.route) badges.appendChild(el("span", "badge route", "route: " + resp.route));
  if (resp.answer_source) badges.appendChild(el("span", "badge", "출처: " + resp.answer_source));
  if (resp.answer_path) badges.appendChild(el("span", "badge", "path: " + resp.answer_path));
  if (resp.confidence) badges.appendChild(el("span", confBadgeClass(resp.confidence), "신뢰도: " + confLabel(resp.confidence)));

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

  // 소요 시간 · 메타
  const kv = $("insp-kv");
  kv.innerHTML = "";
  const addKV = (k, v) => {
    if (v === undefined || v === null || v === "") return;
    kv.appendChild(el("div", "k", k));
    kv.appendChild(el("div", "v", String(v)));
  };
  addKV("소요 시간(초)", resp.elapsed_seconds);
  if (resp.timings && resp.timings.rag_s) addKV("RAG 시간(초)", Number(resp.timings.rag_s).toFixed(1));
  const sm = resp.source_meta || {};
  if (sm.type === "faq") {
    addKV("엑셀 시트", sm.sheet);
    addKV("행", sm.row);
    addKV("질문 유형", sm.question_type);
    addKV("장애 유형", sm.fault_type);
    addKV("유사도", sm.best_score !== undefined ? Number(sm.best_score).toFixed(3) : null);
    if (sm.source_files && sm.source_files.length) addKV("근거 파일", sm.source_files.join(", "));
  }
  if (sm.type === "rag3x") {
    addKV("리랭크 점수", sm.rerank_top_score !== undefined && sm.rerank_top_score !== null ? Number(sm.rerank_top_score).toFixed(4) : null);
    addKV("RAG route", sm.route_reason);
    if (sm.metrics && sm.metrics.timings_seconds) {
      addKV("검색 시간(초)", sm.metrics.timings_seconds.retrieve);
      addKV("생성 시간(초)", sm.metrics.timings_seconds.answer);
    }
  }
  if (sm.type === "web") addKV("provider", sm.provider);
  $("sec-meta").classList.toggle("hidden", kv.children.length === 0);

  // 근거 이미지
  const eviBox = $("insp-evi");
  eviBox.innerHTML = "";
  const evis = (resp.evidence || []).filter((e) => e.image_url);
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
    if (v.transcription_ocr_mismatch) f.push("전사-OCR 불일치");
    flags.appendChild(el("div", "flag-line", "검증: " + (f.length ? f.join(", ") : "이상 없음")));
    hasFlags = true;
  }
  (resp.warnings || []).forEach((w) => { flags.appendChild(el("div", "warn-box", "⚠ " + w)); hasFlags = true; });
  $("sec-flags").classList.toggle("hidden", !hasFlags);
}

// ---- 시나리오 칩 (입력창 위 가로 배열) ----
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

// ---- 통신 ----
function setBusy(on, label) {
  inFlight = on;
  $("busy").classList.toggle("hidden", !on);
  $("btn-send").disabled = on;
  if (on) {
    $("busy-text").textContent = label || "답변 생성 중… (RAG는 25~150초 걸릴 수 있어요)";
    const t0 = Date.now();
    busyTimer = setInterval(() => {
      $("busy-timer").textContent = ((Date.now() - t0) / 1000).toFixed(1) + "s";
    }, 200);
  } else if (busyTimer) {
    clearInterval(busyTimer); busyTimer = null; $("busy-timer").textContent = "";
  }
}

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

async function sendMessage(text) {
  if (inFlight) return;
  addUser(text);
  setBusy(true);
  try {
    const resp = await post("/api/chat", { session_id: sessionId, message: text });
    setSession(resp.session_id);
    addBot(resp);
  } catch (e) {
    handleError(e);
  } finally {
    setBusy(false);
  }
}

async function sendAction(opt) {
  if (inFlight) return;
  addUser("▶ " + opt.label);
  setBusy(true, "시나리오 이동 중…");
  try {
    const resp = await post("/api/chat", {
      session_id: sessionId,
      action: {
        type: "scenario_option",
        scenario_id: opt.scenario_id,
        node_id: opt.node_id,
        option_id: opt.option_id,
        label: opt.label,
      },
    });
    setSession(resp.session_id);
    addBot(resp);
  } catch (e) {
    handleError(e);
  } finally {
    setBusy(false);
  }
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
    $("engine-status").innerHTML =
      "RAG 엔진: <b>" + (eng.status || "?") + "</b>" +
      (eng.error ? " (" + eng.error + ")" : "") +
      " · 백엔드: " + (h.routing && h.routing.backend) +
      " · LangSmith: " + (ls.tracing_enabled ? "on" : "off") +
      " · 웹검색: " + (h.web_search && h.web_search.enabled ? "on" : "off");
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
