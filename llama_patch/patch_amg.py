#!/usr/bin/env python3
"""
patch_amg.py — applies the Adaptive MoE Gate modification to llama-graph.cpp.

Usage:
    python3 patch_amg.py [path/to/llama-graph.cpp]

Default path: ~/llama.cpp/src/llama-graph.cpp
Idempotent: safe to run multiple times, will skip if already patched.
Creates a backup at llama-graph.cpp.bak before modifying.

Fixes vs original:
  - Stats print moved into cb_amg (compute time) — eliminates duplicate lines
  - Slots only counted by thread 0 — fixes thread overcounting
  - Threshold configurable via AMG_THRESHOLD env var (default 0.75, not 0.90)
"""

import sys
import shutil
from pathlib import Path

# ── target file ──────────────────────────────────────────────────────────────
DEFAULT_PATH = Path.home() / "llama.cpp/src/llama-graph.cpp"
target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

if not target.exists():
    print(f"ERROR: file not found: {target}")
    sys.exit(1)

source = target.read_text(encoding="utf-8")

# ── guard: already patched? ───────────────────────────────────────────────────
PATCH_MARKER = "Adaptive MoE Gate"
if PATCH_MARKER in source:
    print("Already patched — nothing to do.")
    sys.exit(0)

# ── backup ────────────────────────────────────────────────────────────────────
backup = target.with_suffix(".cpp.bak")
shutil.copy2(target, backup)
print(f"Backup written to {backup}")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1
# Insert amg_params struct + cb_amg callback before build_moe_ffn.
# Stats print is inside cb_amg (compute time) to avoid duplicate lines.
# Slots counted only by thread 0 to avoid thread overcounting.
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_1 = "ggml_tensor * llm_graph_context::build_moe_ffn("

AMG_CALLBACK = """\
// ── Adaptive MoE Gate ─────────────────────────────────────────────────────
// Zero-gates tail experts past cumulative weight threshold, then renorms.
// Applied after FFN compute — no FLOP savings, quality effect only.
// GPU sync: ggml_map_custom1 runs on CPU -> one GPU->CPU->GPU roundtrip
// per MoE layer per token. Expect ~2-4ms overhead per generated token.
//
// Tunable via environment variables (read once at startup):
//   AMG_THRESHOLD=0.75   cumulative weight threshold (default 0.75)
//   AMG_MIN_K=2          minimum experts always kept (default 2)
//   AMG_DISABLE=1        bypass AMG entirely, use standard top_k
//
// Stats logged to stderr every AMG_LOG_EVERY callback invocations.
//   AMG_LOG_EVERY=500    (default 500, ~10 tokens at 48 layers/token)

struct amg_params {
    float    threshold = 0.75f;
    int      min_k     = 2;
    bool     disabled  = false;
    uint64_t log_every = 500;
};

static amg_params amg_load_params() {
    amg_params p;
    if (const char * v = getenv("AMG_THRESHOLD")) p.threshold = std::stof(v);
    if (const char * v = getenv("AMG_MIN_K"))     p.min_k     = std::stoi(v);
    if (const char * v = getenv("AMG_DISABLE"))   p.disabled  = (std::stoi(v) != 0);
    if (const char * v = getenv("AMG_LOG_EVERY")) p.log_every = (uint64_t)std::stoull(v);
    if (!p.disabled) {
        fprintf(stderr, "[AMG] threshold=%.2f  min_k=%d  log_every=%llu\\n",
                p.threshold, p.min_k, (unsigned long long)p.log_every);
    } else {
        fprintf(stderr, "[AMG] disabled -- standard top_k active\\n");
    }
    return p;
}
static amg_params s_amg = amg_load_params();

// Atomic stats — thread-safe accumulators.
// slots and calls only incremented by thread 0 to avoid overcounting.
static std::atomic<uint64_t> amg_total_slots{0};
static std::atomic<uint64_t> amg_total_used{0};
static std::atomic<uint64_t> amg_call_count{0};

static void cb_amg(ggml_tensor * dst, const ggml_tensor * src,
                   int ith, int nth, void * ud)
{
    (void)nth;
    const auto * p     = static_cast<const amg_params *>(ud);
    const int    min_k = p->min_k;
    const float  thr   = p->threshold;
    const int    max_k = (int)src->ne[0];  // n_expert_used
    const int    n_tok = (int)src->ne[1];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    uint64_t local_used = 0;

    for (int t = ith; t < n_tok; t += nth) {
        const float * in  = (const float *)((const char *)src->data + (size_t)t * src->nb[1]);
        float       * out =       (float *)((      char *)dst->data + (size_t)t * dst->nb[1]);

        float cumsum = 0.0f;
        float kept   = 0.0f;
        bool  done   = false;
        int   active = 0;

        for (int i = 0; i < max_k; i++) {
            if (done) {
                out[i] = 0.0f;
            } else {
                out[i]  = in[i];
                cumsum += in[i];
                kept   += in[i];
                active++;
                if (active >= min_k && cumsum >= thr) done = true;
            }
        }
        if (kept > 1e-8f)
            for (int i = 0; i < max_k; i++) out[i] /= kept;

        local_used += (uint64_t)active;
    }

    // Only thread 0 updates slots and call count to avoid overcounting.
    amg_total_used.fetch_add(local_used, std::memory_order_relaxed);
    if (ith == 0) {
        amg_total_slots.fetch_add((uint64_t)n_tok * (uint64_t)max_k,
                                  std::memory_order_relaxed);
        uint64_t calls = amg_call_count.fetch_add(1, std::memory_order_relaxed) + 1;

        // Periodic stats — fires once per log_every calls, no duplicates.
        if (p->log_every > 0 && calls % p->log_every == 0) {
            uint64_t slots = amg_total_slots.load(std::memory_order_relaxed);
            uint64_t used  = amg_total_used.load(std::memory_order_relaxed);
            if (slots > 0) {
                // avg_active = used / (slots / max_k)  [= used*max_k/slots]
                double avg = (double)used * max_k / (double)slots;
                double pct = 100.0 * avg / max_k;
                fprintf(stderr,
                        "[AMG] avg active experts: %.2f / %d  (%.0f%% of max)"
                        "  calls=%llu\\n",
                        avg, max_k, pct,
                        (unsigned long long)calls);
            }
        }
    }
}
// ──────────────────────────────────────────────────────────────────────────

"""

if ANCHOR_1 not in source:
    print(f"ERROR: anchor not found:\n  {ANCHOR_1}")
    sys.exit(1)

source = source.replace(ANCHOR_1, AMG_CALLBACK + ANCHOR_1, 1)
print("Patch 1 applied: cb_amg callback inserted before build_moe_ffn")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2
# Insert gate invocation before ggml_build_forward_expand(gf, weights).
# No stats print here — that's now in cb_amg.
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_2 = "    //call early so that topk-moe can be used\n    ggml_build_forward_expand(gf, weights);"

AMG_GATE = """\
    // ── Adaptive MoE Gate ────────────────────────────────────────────────
    // weights: [1, n_expert_used, n_tokens], sorted descending, sum=1.
    // Reshape to 2D, apply threshold callback, reshape back.
    if (!s_amg.disabled) {
        ggml_tensor * w2d = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        w2d     = ggml_cont(ctx0, w2d);
        w2d     = ggml_map_custom1(ctx0, w2d, cb_amg, GGML_N_TASKS_MAX, &s_amg);
        cb(w2d, "ffn_moe_weights_amg", il);
        weights = ggml_reshape_3d(ctx0, w2d, 1, n_expert_used, n_tokens);
    }
    // ─────────────────────────────────────────────────────────────────────
"""

if ANCHOR_2 not in source:
    print(f"ERROR: anchor not found:\n  {ANCHOR_2!r}")
    sys.exit(1)

source = source.replace(ANCHOR_2, AMG_GATE + ANCHOR_2, 1)
print("Patch 2 applied: AMG gate block inserted before ggml_build_forward_expand")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3: ensure #include <atomic> is present
# ─────────────────────────────────────────────────────────────────────────────
if "#include <atomic>" not in source:
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines[:30]):
        if line.startswith("#include"):
            insert_at = i + 1
    lines.insert(insert_at, "#include <atomic>\n")
    source = "".join(lines)
    print("Patch 3 applied: #include <atomic> added")
else:
    print("Patch 3 skipped: #include <atomic> already present")

# ── write result ──────────────────────────────────────────────────────────────
target.write_text(source, encoding="utf-8")
print(f"\nDone. Modified file written to {target}")
print("Rebuild: cmake --build ~/llama.cpp/build -j$(nproc) --target llama-server")