# Minesweeper LLM Competition Strategy

## Model: Qwen2.5-14B-Instruct

### Why This Model
- **14B dense parameters**, all active every token (vs MoE models with ~2.5B effective)
- Instruction-tuned: reliable JSON output from day zero
- 32K native context handles all board sizes
- Proven compatibility with Unsloth + TRL GRPO pipeline
- 48 layers, 5120 hidden dim, 40 attention heads for complex constraint reasoning

### Rejected Alternatives
| Model | Reason |
|-------|--------|
| gpt-oss-20b | MoE (32 experts, 4 active) = ~2.5B effective. BASE model with NO chat template |
| Qwen3-4B | Too small for 50x50 frontier constraints. Known GRPO bugs with Unsloth |
| Llama-8B | Less capable than Qwen2.5-14B on structured JSON tasks |
| Gemma-3-12b | Less Unsloth+TRL support |

## Architecture

### 3-Tier Minesweeper Constraint Solver (`solver.py`)

Used for training data generation and reward function deducibility checking.

**Tier 1: Single-Cell Constraint Propagation**
For each revealed number N: count adjacent flags F, unrevealed U.
- N == F -> all U are SAFE
- N - F == |U| -> all U are MINES
- Iterate to fixed point. Coverage: ~60-70%.

**Tier 2: Set-Based Coupled Constraints**
For pairs of numbered cells sharing unrevealed neighbors:
- Subset reduction: if A's variable set subset of B's, subtract -> new constraint
- Coverage: ~85-90%.

**Tier 3: Tank Solver (Backtracking Enumeration)**
1. Frontier identification: unrevealed cells adjacent to revealed numbers
2. Partition into connected components (Union-Find)
3. Backtrack per component: enumerate valid mine configs with constraint pruning
4. Weight by C(Y, M-m): Y=interior cells, M=remaining mines, m=config mines
5. Output: per-cell mine probability. P=0->safe, P=1->mine
- Timeout: 1s per component. Components >35 cells -> fall back to Tier 2

### Prompt Design: Frontier Format for ALL Board Sizes

**Critical Finding**: Compact grid format (showing the full board as ASCII) produced only 10-15% valid moves on boards ≤16x16, while frontier sparse format achieved 100% valid moves on ALL board sizes. The model was trained primarily on frontier format data (~70% of examples) and never generalized to compact grid spatial reasoning.

**FRONTIER_THRESHOLD = 0** (all boards use frontier format)

```
MINESWEEPER 10x10 MINES:15 FLAGS:3 LEFT:12
FRONTIER (numbered cells with hidden neighbors):
R0C2=1 flags:0 hidden:[(0,1)(1,1)(1,2)]
R0C3=2 flags:0 hidden:[(0,4)(1,4)]
...
HIDDEN NEAR NUMBERS: (0,1)(1,1)(1,2)...
TOTAL HIDDEN: 85 INTERIOR(no adj number): 62
RULES: .=hidden F=flag 0-8=adjacent mines
Output ONLY: {"type":"reveal"|"flag","row":R,"col":C}
```

**Why frontier beats compact grid**:
1. Explicit coordinate lists → model picks from listed coordinates → 100% valid targets
2. No spatial reasoning needed → model processes structured constraint data directly
3. Consistent format across all training data → no format generalization gap

### Training Pipeline

**Phase 1: SFT Warmup**
- 50K examples from forward-gameplay solver runs
- 1 epoch, lr=2e-5, cosine schedule
- LoRA rank=64, alpha=128, targets: q,k,v,o,gate,up,down
- Batch size: 2 x 8 gradient accumulation = 16 effective

**Phase 2: GRPO Refinement**
- 1200 steps, DAPO loss with asymmetric clipping (epsilon=0.2/0.28)
- 3 reward functions with weights [1.0, 2.0, 0.5]
- 8 generations per prompt, batch-level reward scaling
- vLLM colocate for fast generation

### Training Dataset (~37K after filtering)

**Board Size Distribution (after 50x50 filtering)**
50x50 examples removed from training: competition max board size < 50x50.
50x50 data had noisiest labels (solver component cap at 35 cells) and longest prompts.
| Size | ~% (post-filter) |
|------|---|
| 6x6 | 6.3% |
| 8x8 | 10.4% |
| 10x10 | 12.4% |
| 16x16 | 22.2% |
| 20x20 | 23.7% |
| 30x30 | 25.1% |

**Game Stage Distribution**
| Stage | % |
|-------|---|
| Opening | 6.6% |
| Early | 11.2% |
| Mid | 38.8% |
| Late | 28.8% |
| Endgame | 14.2% |

**Generation Method:**
- Play games forward using the solver itself (not random mid-game states)
- Random first safe reveal (simulates controller-provided opening move)
- Snapshot at each step as a training example
- Store full mine_positions for robust reconstruction in reward functions
- 94% of examples have deducible optimal moves

### Reward Functions

**1. Format Reward (weight: 1.0)**
- Valid JSON with correct keys: +1.0
- Invalid: -3.0

**2. Gameplay Reward (weight: 2.0)**
- Reveal mine: -1.0 (raw -25/25)
- Out of bounds: -0.6
- Already revealed: -0.48
- Flag non-mine: -0.4
- Flag correct mine: +0.6
- Reveal safe (deducible): +0.6
- Reveal safe (guess): +0.4
- Win: +1.5 (capped to prevent gradient spikes)

**3. Strategic Reward (weight: 0.5)**
- Guessed when deducible move existed: -0.3
- Move adjacent to revealed numbers: +0.2
- Flagged certain mine (flag-first strategy): +0.15
- Over-flagged: -0.4
- Reveal triggers 0-cell cascade (flood-fill): +0.15

### Win Condition
Win requires BOTH: all mines flagged AND all safe cells revealed.
This matches the competition specification exactly.

### GRPO Data Filtering
Prompts exceeding 7500 tokens are filtered from the GRPO dataset to prevent TRL's
max_prompt_length from silently truncating board state, which would cause reward
poisoning (model sees truncated prompt, reward scores against full game state).

### Inference Configuration
- Greedy decoding (temperature=0, do_sample=false)
- max_new_tokens: 64
- No repetition penalty

### Final Model Selection: SFT-only (not GRPO)

GRPO training (400 steps, DAPO loss, num_generations=4) was completed but **degraded performance** compared to SFT-only. The GRPO model scored lower on every board size. Root cause: with only 4 generations per prompt and the SFT model already producing high-quality JSON, reward variance was too low for meaningful GRPO learning (grad_norm=0 on most steps).

**Final model**: SFT-only merged checkpoint at `/workspace/your_finetuned_model`

### Evaluation Results (85 games, 676 moves)

| Board | Games | ValidMove | AvgScore | Notes |
|-------|-------|-----------|----------|-------|
| 6x6 | 20 | 100% | +11.2 | |
| 8x8 | 20 | 100% | +25.2 | |
| 10x10 | 20 | 100% | +43.0 | |
| 16x16 | 10 | 100% | +35.5 | |
| 20x20 | 10 | 100% | +87.0 | Best |
| 30x30 | 5 | 100% | +9.0 | |
| **Total** | **85** | **100%** | **+33.6** | |

### Key Design Decisions
1. **Frontier format for ALL boards (FRONTIER_THRESHOLD=0)**: Most critical decision. Compact grid format produced only 10-15% valid moves; frontier format achieves 100%. The model was trained primarily on frontier format and never generalized to spatial grid reading.
2. **SFT-only model**: GRPO degraded performance due to insufficient reward diversity with 4 generations. SFT alone produces strong results.
3. **Greedy inference**: No sampling noise for structured JSON output
4. **1 epoch SFT**: Prevents memorization on structured output task
5. **Forward-gameplay data**: Solver builds up frontier naturally, guaranteeing solvability
6. **Stage-balanced subsampling**: Prevents late/endgame domination in training data
7. **50x50 filtered from training**: Competition max board size < 50x50; filtering removes noisiest examples
