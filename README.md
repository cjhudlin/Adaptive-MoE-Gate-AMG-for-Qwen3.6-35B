# Adaptive MoE Gate (AMG) for Qwen3.6-35B in llama.cpp — empirical benchmarking of an open research gap

> **TL;DR:** We implemented a post-hoc adaptive expert gating mechanism directly inside llama.cpp for Qwen3.6-35B-A3B (APEX), benchmarked it across four configurations, and found that expanding the router to k=12 with a cumulative threshold gate at 0.90 achieves marginally better perplexity (11.2925) than the stock k=8 baseline (11.3277), while empirically identifying the key limitation — and the path to solving it. The patch script, raw results, and full methodology are included so anyone can reproduce or extend this work.

---

**Model:** Qwen3.6 MoE 35B
**Framework:** llama.cpp (b9833-c818263f2), custom AMG patch
**Date:** June 29, 2026

---

## Introduction

Mixture-of-Experts (MoE) language models like Qwen3.6-35B route each token through a fixed number of expert sub-networks. The number — typically 8 — is hardcoded at training time and used for every token regardless of how semantically simple or complex it is. A common word like "the" activates the same number of experts as a specialised technical term, even though the information requirements are entirely different.

The Adaptive MoE Gate (AMG) is an inference-time modification to llama.cpp that introduces cumulative probability thresholding on expert routing weights. Rather than always using exactly k experts, AMG uses as many experts as needed to reach a confidence threshold — zeroing the weakest contributors and renormalising the survivors. The static computation graph constraint in GGML means all k expert FFNs still execute; AMG adjusts which outputs count, not which computations run.

This report documents the implementation, benchmarking methodology, perplexity results across four configurations on Qwen3.6-35B-A3B, and the empirical finding that explains both why the current results are limited and what is needed to unlock genuine per-token adaptivity.

### What is not new

The concept of threshold-based cumulative routing exists in published literature: XMoE (2024), DynMoE (ICLR 2025), and TopP routing (Huang et al. 2024) all implement the same core idea. All of these train models from scratch with the adaptive mechanism built in.

### What is genuinely new

No published work describes a working `ggml_map_custom1` callback for adaptive gating in a production inference engine. The workaround for the static GGML graph constraint — zero-gating rather than truly dynamic k — is a practically useful engineering contribution. The empirical benchmarking of post-hoc AMG on a production-scale pretrained model (Qwen3.6-35B) is the first such result we are aware of, and it quantifies precisely why the approach is limited without router fine-tuning.

### Why this matters — what genuinely adaptive models would deliver

The ideal implementation selects a dynamic number of experts per token using two parameters working together: a **threshold** (stop adding experts once confidence is sufficient) and a **cap** (hard ceiling to prevent compute overflow). Simple tokens use only as many experts as needed — potentially 2-4. Complex tokens use up to the cap. Compute becomes proportional to content difficulty.

Once a model is trained with an adaptive router, this delivers four concrete benefits:

**1. Compute proportional to difficulty.** A page of simple prose might average 3 active experts per token. A dense technical argument might average 7. The average compute across a real conversation could drop substantially compared to always-k=8, at no quality cost for simple content — saving energy and allowing faster responses on constrained hardware.

**2. Quality ceiling rises for hard tokens.** With a configurable cap (e.g. 12) and an adaptive router, genuinely difficult tokens can draw on more expert knowledge than the original k=8 model ever could. The cap prevents compute from blowing out — complex tokens get up to 12 experts while simple ones get 2-3. The average stays manageable.

**3. Implicit content-aware routing.** An entropy-regularized router learns which tokens are simple vs complex from the training signal itself. The LM loss gradient is weak on predictable tokens and strong on unpredictable ones. The router learns to reflect this — concentrating for predictable tokens, spreading for unpredictable ones. That is genuine semantic adaptivity, not a threshold imposed on a flat distribution.

**4. Graceful degradation under memory pressure.** With adaptive routing and a tunable cap, max_k can be adjusted at runtime to match available VRAM. Lower cap → fewer active experts → still works, just slightly less capable on hard tokens. Fixed k=8 gives no such knob.

> *"The ideal adaptive MoE implementation selects a dynamic number of experts per token — using only as many as needed to reach a confidence threshold, up to a defined maximum cap. This gives simple tokens cheap routing (2-4 experts) and complex tokens full coverage (up to cap), proportioning compute to content difficulty. AMG implements this mechanism at inference time; router fine-tuning is required to make the underlying distributions peaked enough for the threshold to have meaningful bite."*

---

## 1. Background

Qwen3.6-35B-A3B is a Mixture-of-Experts model with 256 total experts per layer, of which exactly 8 are selected per token during inference (fixed top-k routing). The model was pretrained with this fixed-k=8 constraint — the router and expert weights co-evolved assuming exactly 8 experts activate per token.

The AMG was designed to address a theoretical inefficiency: not every token requires the same number of experts. Simple tokens (punctuation, common words) may only need 2-3 experts to be well-represented, while complex reasoning tokens may benefit from all 8 or more. Fixed top-k always uses exactly k experts regardless of token complexity.

**AMG mechanism:** After the router selects the top-k experts and normalises their weights to sum to 1.0, a CPU callback zeros out tail experts whose cumulative weight contribution falls below a threshold, then renormalises the survivors. This preserves the static GGML computation graph (all k expert FFNs still compute) while dynamically adjusting which expert outputs contribute to the final result.

---

## 2. Implementation

The patch is applied to `src/llama-graph.cpp` in llama.cpp via the included `patch_amg.py` script, which is idempotent and creates a backup before modifying. Two components are added:

**Component 1 — `cb_amg` callback** (inserted before `llm_graph_context::build_moe_ffn`):
- Iterates over sorted expert weights (descending) per token
- Accumulates cumulative weight until threshold is reached
- Zeros remaining experts, renormalises survivors
- Tracks atomic stats for logging (thread-safe, thread-0 only for counters)

**Component 2 — Gate insertion** (inside `build_moe_ffn`, before `ggml_build_forward_expand`):
- Reshapes weights tensor to 2D for the callback
- Applies `ggml_map_custom1` (CPU backend)
- Reshapes result back to 3D

**Runtime configuration via environment variables:**

| Variable | Default | Effect |
|---|---|---|
| `AMG_THRESHOLD` | 0.75 | Cumulative weight threshold |
| `AMG_MIN_K` | 2 | Minimum experts always kept |
| `AMG_DISABLE` | 0 | Set to 1 to disable entirely |
| `AMG_LOG_EVERY` | 500 | Callback invocations between log lines |

**GPU sync caveat:** `ggml_map_custom1` executes on the CPU backend. For each MoE layer per token, this triggers one GPU→CPU→GPU data transfer of the weight tensor. With 40 MoE layers in Qwen3.6-35B, this is 40 roundtrips per generated token, adding approximately 2-4ms overhead per token at typical generation speeds.

**To apply and build:**
```bash
python3 patch_amg.py ~/llama.cpp/src/llama-graph.cpp
cmake --build ~/llama.cpp/build -j$(nproc) --target llama-server
```

**To produce the k12 GGUF:**
```bash
llama-quantize --override-kv qwen35moe.expert_used_count=int:12 \
    original.gguf k12-fixed.gguf copy
```

> **Critical note:** The correct GGUF metadata key is `qwen35moe.expert_used_count`, not `qwen3moe.expert_used_count`. Using the wrong key silently fails — the override is written but ignored at load time. Verify with:
> ```bash
> python3 -c "
> import sys; sys.path.insert(0, 'gguf-py')
> from gguf import GGUFReader
> r = GGUFReader('model.gguf')
> [print(k, r.fields[k].parts[-1].tolist()) for k in r.fields if 'expert' in k.lower()]
> "
> ```

Run perplexity (on the K12 model for example) (in WSL)
AMG_THRESHOLD=0.90 ./build/bin/llama-perplexity \
    -m /path/to/model/qwen3-35b-a3b-k12.gguf \
    -f ptb.test.txt \
    --ctx-size 512 \
    -ot "ffn_.*_exps=CPU"

---

## 3. Experiment Design

Four configurations were benchmarked using `llama-perplexity` on the Penn Treebank test set (PTB, 192 chunks, ctx=512):

| Config | Model | AMG | Threshold | Expert slots | Expert FFNs computed |
|---|---|---|---|---|---|
| **Baseline** | k8 original | OFF | — | 8 | 8 |
| **AMG-k8-0.75** | k8 original | ON | 0.75 | 8 | 8 |
| **AMG-k12-OFF** | k12 patched | OFF | — | 12 | 12 |
| **AMG-k12-0.90** | k12 patched | ON | 0.90 | 12 | 12 |

All runs used `-ot "ffn_.*_exps=CPU"` to offload expert weight tensors to RAM, required because the full model exceeds the 16GB VRAM budget when all expert weights are on GPU.

---

## 4. Results

### 4.1 Perplexity

| Config | PPL | ±Error | vs Baseline | Within error? |
|---|---|---|---|---|
| **Baseline** (k8, AMG OFF) | 11.3277 | ±0.143 | — | — |
| AMG-k8-0.75 | 12.1226 | ±0.155 | **+0.795** | ❌ No — clear degradation |
| AMG-k12-OFF | 11.3379 | ±0.144 | +0.010 | ✅ Yes — negligible |
| **AMG-k12-0.90** | **11.2925** | **±0.143** | **−0.035** | ✅ Yes — marginal improvement |

Error ranges (non-overlapping ranges indicate statistically significant differences):

| Config | Low | High |
|---|---|---|
| Baseline | 11.185 | 11.471 |
| AMG-k8-0.75 | 11.968 | 12.278 |
| AMG-k12-OFF | 11.194 | 11.482 |
| AMG-k12-0.90 | 11.150 | 11.436 |

### 4.2 Expert Utilisation

| Config | Avg active | Max slots | Utilisation |
|---|---|---|---|
| Baseline | 8.00 | 8 | 100% |
| AMG-k8-0.75 | 5.42 | 8 | 67.8% |
| AMG-k12-OFF | 12.00 | 12 | 100% |
| AMG-k12-0.90 | 10.31 | 12 | 85.9% |

### 4.3 Runtime (perplexity benchmark, 192 chunks)

| Config | Time | vs Baseline |
|---|---|---|
| Baseline (k8, AMG OFF) | ~7:45 | — |
| AMG-k8-0.75 | ~7:58 | +1.7% |
| AMG-k12-OFF | ~8:54 | +14.8% |
| AMG-k12-0.90 | ~9:17 | +19.7% |

---

## 5. Analysis

### 5.1 AMG-k8-0.75 — failed configuration

Cutting from 8 to an average of 5.42 active experts at threshold 0.75 produced a clear, statistically significant PPL degradation of +0.795. The error ranges do not overlap at all ([11.97–12.28] vs [11.19–11.47]).

The root cause is the training distribution mismatch. The model's router was trained to distribute weight across exactly 8 experts per token, producing relatively flat distributions:
```
Typical post-norm_w distribution (8 experts):
[0.16, 0.14, 0.13, 0.12, 0.12, 0.11, 0.11, 0.11]
Cumulative sum at k=5: ~0.67 — below 0.75 threshold
Cumulative sum at k=7: ~0.89 — above 0.75 threshold
```
To cut 2-3 experts at this threshold, you must cross into experts that genuinely contribute ~11-13% of the signal each. This is a real quality loss, not noise suppression.

### 5.2 AMG-k12-OFF — the cost of extra experts

Forcing 12 expert FFN computations per token yields PPL = 11.3379, essentially identical to baseline (difference 0.010, well within ±0.143 error). The 4 extra experts produce marginal signal that on average neither helps nor hurts quality. Runtime increases by ~15% due to additional FFN computations.

This confirms the model can tolerate k>8 without catastrophic behaviour, but the extra computation is largely wasted when all 12 outputs are summed equally.

### 5.3 AMG-k12-0.90 — the most promising result

By selecting up to 12 experts but gating the weakest ones with threshold 0.90, the model achieves PPL = 11.2925 — 0.035 below the k8 baseline. While this improvement is within the statistical error margin and cannot be claimed as definitively significant, the direction is consistent:

- The model uses an average of 10.31 out of 12 expert slots
- Approximately 1.69 experts are zeroed per token on average
- The zeroed experts are those with the lowest routing weight — likely least relevant for each token
- Cutting these and renormalising the survivors slightly amplifies the retained expert contributions

The intuition: with k=12, the model selects 4 experts it would never have seen during training. These 4 extras have genuinely low routing weight. AMG at 0.90 removes the weakest 1-2 of these, leaving the 10 strongest — a cleaner signal than including all 12 equally weighted.

### 5.4 Why the distribution is still flat

The AMG log shows `10.31 / 12` consistently across all 192 chunks, with almost no variation. The absence of per-token variation is the expected consequence of training: the router learned to spread weight across k slots. With k=12, it spreads across 12 instead of 8 — still flat, just wider. Genuine per-token adaptivity (simple tokens using 2-4 experts, complex tokens using 8-12) requires a router trained with entropy regularization, not inference-time patching alone.

---

## 6. Key Findings

**Finding 1:** Post-hoc AMG on a fixed-k trained model cannot produce meaningful per-token expert variability without quality degradation. The router produces flat distributions by design.

**Finding 2:** The correct GGUF metadata key for this model is `qwen35moe.expert_used_count`, not `qwen3moe.expert_used_count`. Using the wrong key silently fails — the override is written to the file but ignored at load time.

**Finding 3:** k12 with AMG at threshold 0.90 achieves the best PPL of all configurations tested (11.2925), marginally below the k8 baseline. While not statistically conclusive at one standard error, it is the most practically useful configuration available without retraining.

**Finding 4:** k12 without AMG adds ~15% compute overhead with negligible quality change. k12 with AMG adds ~20% overhead but achieves the lowest PPL. The quality-per-FLOP trade-off favours AMG-k12-0.90 over plain k12.

**Finding 5:** Threshold-based cumulative routing has been studied in the research literature (TopP routing, XMoE, DynMoE) but all successful implementations train models from scratch with the adaptive mechanism. Post-hoc application to pretrained fixed-k models is an open research gap that this work quantifies empirically for the first time on a production-scale model.

---

## 7. Recommended Configuration

```bash
# Produce k12 GGUF (one-time)
llama-quantize --override-kv qwen35moe.expert_used_count=int:12 \
    original.gguf qwen36-35b-k12.gguf copy

# Run with AMG at 0.90 threshold
AMG_THRESHOLD=0.90 AMG_MIN_K=2 ./llama-server \
    -m qwen36-35b-k12.gguf \
    --host 0.0.0.0 --port 8081 \
    -ot "ffn_.*_exps=CPU" \
    --jinja

# Monitor expert utilisation
journalctl -u llama-server -f | grep AMG
```

---

## 8. Next Steps — Router Fine-tuning

To achieve genuine per-token variability, the router layers (`mlp.gate.weight`, 40 layers × [256, 2048] = 21M parameters) need fine-tuning with an entropy regularization loss that rewards peaked routing distributions for simple tokens:

```
L = L_LM + λ_entropy × H(router) + λ_balance × KL(usage, uniform)
```

A router fine-tuning pipeline (`router_train.py`) was developed targeting only these 21M parameters with all other model weights frozen. Hardware requirement: ~20GB VRAM (A100 40GB recommended). The pipeline is included in this repository.

Success criterion: AMG-k12-0.90 PPL ≤ 11.33 (within baseline error) with genuine per-token variance in active expert count (σ > 1.5 across tokens), rather than the current near-constant 10.31.

---

## Appendix: Raw Results

| Config | Chunk 1 | Chunk 50 | Chunk 100 | Chunk 150 | Final PPL | ±Error |
|---|---|---|---|---|---|---|
| k8 AMG OFF | 12.155 | 10.682 | 11.037 | 11.306 | 11.3277 | ±0.143 |
| k8 AMG 0.75 | 12.827 | 11.352 | 11.795 | 12.086 | 12.1226 | ±0.155 |
| k12 AMG OFF | 12.227 | 10.732 | 11.083 | 11.305 | 11.3379 | ±0.144 |
| k12 AMG 0.90 | 12.033 | 10.679 | 11.045 | 11.276 | 11.2925 | ±0.143 |

Raw perplexity logs and the patch script are included alongside this document.
