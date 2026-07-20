"""P2 스모크 테스트 — Gemini Flash 연결/키/모델/토큰·비용/한도 확인 (키 비노출).

- .env 키 로드(값 미출력, 존재 여부·길이만).
- generateContent 1회(짧은 프롬프트, thinking=0) → 응답·usageMetadata·지연 측정.
- 429/한도 헤더 확인.
"""
import io, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
TEST3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST3))

import requests
from rag3x.gemini_backend import _load_api_key, _API_HOST, _PRICE_IN_PER_M, _PRICE_OUT_PER_M

key = _load_api_key()
print(f"[key] 로드 OK (길이 {len(key)}, 값 비노출)")

MODEL = "gemini-2.5-flash"
url = f"{_API_HOST}/{MODEL}:generateContent"
payload = {
    "contents": [{"role": "user", "parts": [{"text": "다음 근거만 보고 답해라. 근거: '서울의 인구는 940만명이다.' 질문: 서울 인구는? 한 문장으로."}]}],
    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512, "thinkingConfig": {"thinkingBudget": 0}},
}

t0 = time.time()
r = requests.post(url, params={"key": key}, json=payload, timeout=60,
                  headers={"Content-Type": "application/json"})
dt = time.time() - t0
print(f"[http] status={r.status_code}  지연={dt:.2f}s")
# 한도 관련 헤더(있으면)
for h in ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining"):
    if h in r.headers:
        print(f"[hdr] {h}={r.headers[h]}")

if r.status_code != 200:
    body = r.text[:300]
    # 키가 본문/URL에 반사되지 않는지 안전 확인
    print(f"[http] 실패 본문(앞300자): {body.replace(key, '***REDACTED***')}")
    sys.exit(1)

data = r.json()
cand = data["candidates"][0]
text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])).strip()
u = data.get("usageMetadata", {})
pin, pout, ptot = u.get("promptTokenCount"), u.get("candidatesTokenCount"), u.get("totalTokenCount")
cost = (pin or 0) / 1e6 * _PRICE_IN_PER_M + (pout or 0) / 1e6 * _PRICE_OUT_PER_M
print(f"[resp] {text!r}")
print(f"[usage] in={pin} out={pout} total={ptot}  thinking={u.get('thoughtsTokenCount')}")
print(f"[cost] 이 호출 ${cost:.6f} (요율 가정 in ${_PRICE_IN_PER_M}/M, out ${_PRICE_OUT_PER_M}/M)")
print("SMOKE OK")
