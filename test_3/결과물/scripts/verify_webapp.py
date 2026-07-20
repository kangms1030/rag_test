"""웹 데모 검증: /api/stats, /api/ask(정답+이미지 200), 처리 중 중복 요청 429, GET / HTML."""
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = "http://127.0.0.1:8017"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, r.read()


def post_json(path, obj, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


st, body = get("/")
assert st == 200 and b"RAG3" in body, "index.html 서빙 실패"
print("GET / OK")

st, stats = post_json("/api/warmup", {}) if False else get("/api/stats")
stats = json.loads(stats)
assert stats["benchmark"]["rows"] and stats["corpus"]["chunks"] == 2562
print("GET /api/stats OK:", stats["corpus"]["documents"], "docs")

q = "스쿨넷 5단계 재해복구(DR)서비스 요금체계에서 제주특별자치도교육청의 100M와 500M 월 요금은 각각 얼마인가요?"
result = {}
busy_status = {}


def ask_main():
    st, r = post_json("/api/ask", {"question": q})
    result["st"], result["r"] = st, r


t = threading.Thread(target=ask_main)
t.start()
time.sleep(5)  # 처리 중일 때 중복 요청
try:
    post_json("/api/ask", {"question": "테스트"}, timeout=15)
    busy_status["code"] = 200
except urllib.error.HTTPError as e:
    busy_status["code"] = e.code
t.join()

assert busy_status["code"] == 429, f"동시 요청이 429가 아님: {busy_status}"
print("동시 요청 429 OK")

r = result["r"]
assert result["st"] == 200
assert "2,907,000" in r["final_answer"] and r["confidence"] == "high", r["final_answer"][:120]
print("답변 OK:", r["final_answer"][:60].replace("\n", " "), "...")
print("지표: elapsed", r["elapsed_seconds"], "s, timings", r["trace"]["timings"],
      ", LLM", r["trace"]["llm_total"], "회, rerank_top", r["rerank_top_score"])

evs = [e for e in r["evidence"] if e["image_url"]]
assert evs, "근거 이미지 URL 없음"
for e in evs:
    st, img = get(e["image_url"])
    assert st == 200 and img[:8] == b"\x89PNG\r\n\x1a\n", f"이미지 서빙 실패: {e['image_url']}"
    print(f"이미지 200 OK: {e['image_url']} ({len(img)//1024}KB, p{e['page_number']}, score {e['page_score']})")

assert r["trace"]["rows"][0]["stage"].startswith("S1")
print("WEBAPP VERIFY OK")
