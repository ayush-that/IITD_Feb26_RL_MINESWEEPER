# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minesweeper LLM competition entry: a fine-tuned Qwen2.5-14B-Instruct model that plays Minesweeper. The model outputs JSON actions (`{"type":"reveal"|"flag","row":R,"col":C}`) given a board state in frontier format.

**Final model**: SFT-only (no GRPO) merged at `/workspace/your_finetuned_model` (27.5 GB, 16-bit).

## Commands

```bash
# Generate training data (50K examples, ~90s with 32 workers)
python generate_data.py --target 50000 --workers 32 --output minesweeper_training_data.jsonl --frontier-threshold 16

# Training (SFT + GRPO pipeline)
python minesweeper_train.py

# Evaluation
python eval_final.py        # Full evaluation across board sizes
python eval_compare.py      # SFT vs GRPO comparison
python test_e2e.py          # End-to-end test via agent code path

# Agent server (persistent, watches inputs/game_state.json → outputs/action.json)
python -m agents.agent_server --config minesweeper_config.yaml
```

## Environment

- **Hardware**: AMD MI300X, 256GB VRAM, ROCm
- **Critical**: `export VLLM_USE_TRITON_FLASH_ATTN=0` for Qwen2.5 SWA on ROCm
- HF cache at `/root/.cache/huggingface/` is **read-only**

## Architecture

### Data Flow
```
generate_data.py (multiprocessing) → solver.py (3-tier) → training_data.jsonl
→ minesweeper_train.py (SFT) → your_finetuned_model/
→ agents/ (inference server) → action.json
```

### Solver (`solver.py`) — 3-Tier Constraint Satisfaction
- **Tier 1**: Single-cell propagation (N==F → safe, N-F==|U| → mines). ~60-70% coverage.
- **Tier 2**: Set-based coupled constraints (subset reduction on paired cells). ~85-90% coverage.
- **Tier 3**: Tank solver — backtracking enumeration over frontier components (Union-Find partitioning), weighted by `C(Y, M-m)` using lgamma (not `math.comb`, which overflows). 1s timeout, 35-cell component cap.

### Data Generation (`generate_data.py`)
Forward-gameplay: plays games with the solver, snapshots at each step. Stage-balanced subsampling prevents late/endgame domination. 50x50 boards filtered from training data.

### Training (`minesweeper_train.py`)
SFT with LoRA (rank=64, alpha=128) targeting q,k,v,o,gate,up,down. 1 epoch, lr=2e-5, batch=16 effective. GRPO was attempted but degraded performance (insufficient reward diversity with 4 generations → grad_norm=0).

### Agents (`agents/`)
- `minesweeper_agent.py` — Prompt builder. **FRONTIER_THRESHOLD=0**: frontier format for ALL boards (compact grid only got 10-15% valid moves; frontier gets 100%).
- `minesweeper_model.py` — Model loader, greedy inference (temperature=0, max_new_tokens=64).
- `agent_server.py` — Persistent server, loads model once, watches for game states, atomic writes.

## Critical Design Decisions

1. **Frontier format for ALL boards** (`FRONTIER_THRESHOLD=0`): The model never learned to read ASCII grids spatially. Frontier format lists coordinates explicitly → 100% valid moves.
2. **SFT-only**: GRPO degraded performance. With 4 generations/prompt and already-good SFT output, reward variance was too low.
3. **Greedy decoding**: No sampling for structured JSON output.
4. **lgamma for binomial weights**: `math.comb()` overflows for large interior cell counts.

## Known Issues / Workarounds

- Unsloth `save_pretrained_merged` crashes → use PEFT `merge_and_unload()` + save on CPU
- GRPO `use_vllm=False` required (LoRA model lacks `vllm_engine` attribute)
- GRPO `num_generations=8` OOMs on MI300X with 14B → use 4
- SFTTrainer `formatting_func` must return list of strings (not single string)
- TRL dataset "messages" field must be list of dicts, not JSON strings
