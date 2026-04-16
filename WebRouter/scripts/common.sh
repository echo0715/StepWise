#!/usr/bin/env bash

find_free_gpus() {
    local n_needed="${1:-1}"
    local mem_threshold_mb="${2:-${FREE_GPU_MEMORY_THRESHOLD_MB:-500}}"
    local sleep_seconds="${3:-${FREE_GPU_POLL_SECONDS:-20}}"
    local max_attempts="${4:-${FREE_GPU_MAX_ATTEMPTS:-5400}}"
    local allowed_gpus_csv="${5:-${FREE_GPU_ALLOWLIST:-}}"
    local max_used_fraction="${6:-${FREE_GPU_MAX_USED_FRACTION:-}}"
    local normalized_allowed_gpus=""
    local scope_desc="all GPUs"
    local selection_desc="memory.used < ${mem_threshold_mb} MiB"
    local attempt=0

    if [ -n "$allowed_gpus_csv" ]; then
        normalized_allowed_gpus=",${allowed_gpus_csv//[[:space:]]/},"
        scope_desc="GPUs ${allowed_gpus_csv// /}"
    fi
    if [ -n "$max_used_fraction" ]; then
        selection_desc="memory.used / memory.total <= ${max_used_fraction}"
    fi

    while [ "$attempt" -lt "$max_attempts" ]; do
        local free_gpus=()
        while IFS=',' read -r idx mem_used mem_total; do
            idx=$(echo "$idx" | xargs)
            mem_used=$(echo "$mem_used" | xargs | sed 's/ MiB//')
            mem_total=$(echo "$mem_total" | xargs | sed 's/ MiB//')

            if [ -z "$idx" ]; then
                continue
            fi
            if [ -n "$normalized_allowed_gpus" ] && [[ "$normalized_allowed_gpus" != *",$idx,"* ]]; then
                continue
            fi
            if [ -n "$max_used_fraction" ]; then
                if awk "BEGIN { exit !(($mem_used / $mem_total) <= $max_used_fraction) }"; then
                    free_gpus+=("$idx")
                fi
            elif [ "$mem_used" -lt "$mem_threshold_mb" ]; then
                free_gpus+=("$idx")
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)

        if [ "${#free_gpus[@]}" -ge "$n_needed" ]; then
            local result=""
            local i
            for i in $(seq 0 $((n_needed - 1))); do
                [ -n "$result" ] && result+=","
                result+="${free_gpus[$i]}"
            done
            echo "$result"
            return 0
        fi

        attempt=$((attempt + 1))
        echo "$(date): Need $n_needed free GPUs from $scope_desc ($selection_desc), found ${#free_gpus[@]}. Retrying in ${sleep_seconds}s ($attempt/$max_attempts)..." >&2
        sleep "$sleep_seconds"
    done

    echo "ERROR: No free GPU found after $max_attempts attempts" >&2
    return 1
}

wait_for_vllm() {
    local target="$1"
    local max_wait="${2:-300}"
    local models_url=""

    if [[ "$target" =~ ^[0-9]+$ ]]; then
        models_url="http://localhost:${target}/v1/models"
    else
        models_url="${target%/}/models"
    fi

    echo "Waiting for vLLM on $models_url (max ${max_wait}s)..."
    local i
    for i in $(seq 1 "$max_wait"); do
        if curl -s "$models_url" > /dev/null 2>&1; then
            echo "vLLM ready!"
            curl -s "$models_url" | python3 -m json.tool 2>/dev/null || true
            return 0
        fi
        if [ -n "${VLLM_PID:-}" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM process died"
            return 1
        fi
        sleep 1
    done

    echo "ERROR: vLLM did not start in ${max_wait}s"
    return 1
}

get_vllm_served_models() {
    local target="$1"
    local models_url=""

    if [[ "$target" =~ ^[0-9]+$ ]]; then
        models_url="http://localhost:${target}/v1/models"
    else
        models_url="${target%/}/models"
    fi

    python3 - "$models_url" <<'PY'
import json
import sys
import urllib.request

models_url = sys.argv[1]
with urllib.request.urlopen(models_url, timeout=5) as resp:
    data = json.load(resp)

for item in data.get("data", []):
    if isinstance(item, dict):
        model_id = item.get("id")
        if model_id:
            print(model_id)
PY
}

reset_webarena_verified_env() {
    local project_root="${1:-}"
    local docker_root="${2:-}"
    local init_wait_seconds="${3:-${WEB_ARENA_RESET_WAIT_SECONDS:-120}}"
    local failed=0

    if [ -z "$project_root" ]; then
        echo "ERROR: reset_webarena_verified_env requires project_root" >&2
        return 1
    fi
    if [ -z "$docker_root" ]; then
        docker_root="$project_root/third_party/webarena-verified"
    fi

    echo "Resetting WebArena Verified Docker containers..."
    docker stop \
        webarena-verified-shopping webarena-verified-shopping_admin \
        webarena-verified-reddit webarena-verified-gitlab 2>/dev/null || true
    docker rm \
        webarena-verified-shopping webarena-verified-shopping_admin \
        webarena-verified-reddit webarena-verified-gitlab 2>/dev/null || true

    (
        cd "$docker_root"
        docker compose up -d shopping shopping_admin reddit gitlab
    )

    echo "Waiting ${init_wait_seconds} seconds for WebArena services to initialize..."
    sleep "$init_wait_seconds"
    echo "Verifying WebArena services..."

    for svc in \
        "Shopping|http://localhost:7770|200" \
        "Shopping Admin|http://localhost:7780|200" \
        "Reddit|http://localhost:9999|200" \
        "GitLab|http://localhost:8023|302"
    do
        IFS='|' read -r name url expected <<< "$svc"
        code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
        if [ "$code" = "$expected" ] || [ "$code" = "200" ] || [ "$code" = "302" ]; then
            echo "  [OK] $name - HTTP $code"
        else
            echo "  [FAIL] $name - HTTP $code"
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "WebArena Docker reset failed verification" >&2
        return 1
    fi

    echo "WebArena Docker reset complete."
}
