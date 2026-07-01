#!/usr/bin/env bash
# Functional smoke tests for xyz-model-training-pipeline-service (port 8110)
# Usage: ./test-model-training-pipeline-service-api.sh [BASE_URL]
# Exit code: 0 = all passed, 1 = one or more failures

BASE_URL="${1:-http://localhost:8110}"
PASS=0; FAIL=0; SKIP=0
TENANT_ID="00000000-0000-0000-0000-000000000001"
USER_ID="00000000-0000-0000-0000-000000000002"

AUTH_HEADERS=(
  -H "x-verified-tenant-id: ${TENANT_ID}"
  -H "x-verified-user-id: ${USER_ID}"
  -H "x-verified-roles: PLATFORM_ADMIN"
  -H "x-verified-platform: true"
)

log_pass() { PASS=$((PASS+1)); echo "  ✅ $1" >&2; }
log_fail() { FAIL=$((FAIL+1)); echo "  ❌ $1" >&2; }
log_skip() { SKIP=$((SKIP+1)); echo "  ⏭  $1" >&2; }
log_section() { echo "" >&2; echo "=== $1 ===" >&2; }

call() {
  local desc="$1" expected_status="$2"; shift 2
  local response; response=$(curl -s -w "\n%{http_code}" "$@")
  local body; body=$(echo "$response" | head -n -1)
  local status_code; status_code=$(echo "$response" | tail -1)
  if [ "$status_code" = "$expected_status" ]; then
    log_pass "[$status_code] $desc"
  else
    log_fail "[$status_code ≠ $expected_status] $desc — body: $(echo "$body" | head -c 200)"
  fi
  echo "$body"
}

CREATED_RUN_ID=""

log_section "TC-HEALTH — Health & Readiness"
HEALTH_BODY=$(call "TC-HEALTH-01: GET /v1/health returns 200" 200 "${AUTH_HEADERS[@]}" "${BASE_URL}/v1/health")
if echo "${HEALTH_BODY}" | python -c "import sys,json; d=json.load(sys.stdin); assert 'status' in d and 'checks' in d" 2>/dev/null; then
  log_pass "TC-HEALTH-02: health response has status and checks fields"
else
  log_fail "TC-HEALTH-02: health response missing required fields"
fi

log_section "TC-AUTH — Authentication enforcement"
call "TC-AUTH-01: POST /v1/training/runs no headers → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -d '{"trigger_type":"MANUAL","reason":"test"}' \
  "${BASE_URL}/v1/training/runs"
call "TC-AUTH-02: GET /v1/training/runs no headers → 401" 401 \
  "${BASE_URL}/v1/training/runs"
call "TC-AUTH-03: GET /v1/models no headers → 401" 401 \
  "${BASE_URL}/v1/models"
call "TC-AUTH-04: POST /v1/models/1/promote no headers → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -d '{"target_stage":"staging"}' \
  "${BASE_URL}/v1/models/1/promote"

log_section "TC-TRAINING — Trigger and track training runs"
RUN_BODY=$(call "TC-TR-01: POST /v1/training/runs MANUAL → 202" 202 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"trigger_type":"MANUAL","reason":"audit-test"}' \
  "${BASE_URL}/v1/training/runs")
if echo "${RUN_BODY}" | python -c "import sys,json; d=json.load(sys.stdin); assert 'run_id' in d and d['status'] in ('PENDING','RUNNING','COMPLETED','FAILED')" 2>/dev/null; then
  log_pass "TC-TR-02: response has run_id and valid status"
  CREATED_RUN_ID=$(echo "${RUN_BODY}" | python -c "import sys,json; print(json.load(sys.stdin)['run_id'])" 2>/dev/null || echo "")
else
  log_fail "TC-TR-02: response missing run_id or invalid status"
fi

call "TC-TR-03: POST /v1/training/runs invalid trigger_type → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"trigger_type":"BOGUS","reason":"test"}' \
  "${BASE_URL}/v1/training/runs"

if [ -n "${CREATED_RUN_ID}" ]; then
  GET_BODY=$(call "TC-TR-04: GET /v1/training/runs/{run_id} → 200" 200 \
    "${AUTH_HEADERS[@]}" "${BASE_URL}/v1/training/runs/${CREATED_RUN_ID}")
  if echo "${GET_BODY}" | python -c "import sys,json; d=json.load(sys.stdin); assert d['run_id']==\"${CREATED_RUN_ID}\"" 2>/dev/null; then
    log_pass "TC-TR-05: GET run returns correct run_id"
  else
    log_fail "TC-TR-05: GET run returned wrong or missing run_id"
  fi
else
  log_skip "TC-TR-04: skip GET run — no run_id from create"
  log_skip "TC-TR-05: skip run_id verification"
fi

call "TC-TR-06: GET /v1/training/runs/{run_id} non-existent → 404" 404 \
  "${AUTH_HEADERS[@]}" "${BASE_URL}/v1/training/runs/00000000-0000-0000-0000-000000000099"

log_section "TC-RUNS-LIST — List training runs"
RUNS_LIST=$(call "TC-LIST-01: GET /v1/training/runs → 200" 200 \
  "${AUTH_HEADERS[@]}" "${BASE_URL}/v1/training/runs")
if echo "${RUNS_LIST}" | python -c "import sys,json; d=json.load(sys.stdin); assert isinstance(d, list)" 2>/dev/null; then
  log_pass "TC-LIST-02: response is a list"
else
  log_fail "TC-LIST-02: response is not a list"
fi

call "TC-LIST-03: GET /v1/training/runs?limit=200 exceeds max → 422" 422 \
  "${AUTH_HEADERS[@]}" "${BASE_URL}/v1/training/runs?limit=200"

log_section "TC-MODELS — Model listing and champion"
MODELS_LIST=$(call "TC-MOD-01: GET /v1/models → 200" 200 \
  "${AUTH_HEADERS[@]}" "${BASE_URL}/v1/models")
if echo "${MODELS_LIST}" | python -c "import sys,json; d=json.load(sys.stdin); assert isinstance(d, list)" 2>/dev/null; then
  log_pass "TC-MOD-02: models response is a list"
else
  log_fail "TC-MOD-02: models response is not a list"
fi

log_section "TC-PROMOTE — Model promotion"
call "TC-PROM-01: POST /v1/models/1/promote invalid stage → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"target_stage":"invalid_stage"}' \
  "${BASE_URL}/v1/models/1/promote"

log_section "TC-ROLLBACK — Emergency rollback"
call "TC-ROLL-01: POST /v1/models/rollback empty reason → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"reason":""}' \
  "${BASE_URL}/v1/models/rollback"

echo "" >&2
echo "================================" >&2
echo "PASS:${PASS}  FAIL:${FAIL}  SKIP:${SKIP}" >&2
echo "================================" >&2
[ "$FAIL" -eq 0 ]
