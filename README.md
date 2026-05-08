# Does Optimization Parallel Inference Change LLM Inference Outputs?

This project investigates whether system-level optimization configurations in LLM inference engines, specifically vLLM, produce different outputs given the same model and inputs. We test 10 different configurations on a single GPU and compare outputs using multiple metrics.

## Research Question

Given the same model and input, do different system-level inference configurations produce the same output?

Sub-questions:

- Does chunked prefill change outputs?
- Does CUDA graph execution change outputs?
- Does KV cache prefix caching change outputs?
- Does batch processing change outputs?
- Does GPU memory pressure change outputs?
- Does FP8 KV cache quantization change outputs?
- Does CPU swap/offloading change outputs?
- Is the baseline output deterministic across runs?

## Setup

```bash
# Clone and install
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v
```

Note: Requires NVIDIA GPU with CUDA support and vLLM-compatible drivers.

## Usage

**Run all experiments:**

```bash
bash scripts/run_all.sh
```

**Run with custom settings:**

```bash
MODEL=Qwen/Qwen2-1.5B-Instruct MAX_TOKENS=256 bash scripts/run_all.sh
```

**Run a single configuration:**

```bash
python scripts/run_experiment.py --config c0_baseline
```

**Generate report from existing results:**

```bash
python scripts/run_experiment.py --report-only
```

**Full CLI options:**

```
--config NAME         Run a single config by name
--all                 Run all configurations
--model MODEL         Model name (default: Qwen/Qwen2-1.5B-Instruct)
--max-model-len N     Max context length (default: 2048)
--max-tokens N        Max tokens to generate (default: 256)
--output-dir DIR      Raw results directory (default: results/raw)
--report-dir DIR      Reports directory (default: results/reports)
--skip-unsupported    Skip optional configs that fail (default: True)
--report-only         Generate report from existing results
--include-semantic    Include semantic similarity (slow)
--log-level LEVEL     Logging level (default: INFO)
```

## Experiment Configurations

| ID | Name | What It Tests | Key Parameter |
|----|------|---------------|---------------|
| C0 | Baseline | Reference point (no optimizations) | enforce_eager=True, no chunked prefill |
| C1 | Chunked Prefill (512) | Execution chunking | max_num_batched_tokens=512 |
| C2 | Chunked Prefill (2048) | Execution chunking (larger) | max_num_batched_tokens=2048 |
| C3 | CUDA Graphs | Kernel execution path | enforce_eager=False |
| C4 | Prefix Caching | KV cache reuse | enable_prefix_caching=True |
| C5 | Batch Processing | Parallel scheduling | Prompts run as batch |
| C6 | Low GPU Memory | Memory pressure | gpu_memory_utilization=0.4 |
| C7 | FP8 KV Cache | KV cache precision | kv_cache_dtype=fp8_e5m2 (optional) |
| C8 | CPU Swap | KV cache location | swap_space=4 |
| C9 | Determinism Check | Repeatability | 3 runs of baseline |

## Evaluation Metrics

The project uses these metrics for comparison:

- **Exact Match**: Binary, are outputs character-for-character identical?
- **Token Match Ratio**: Fraction of tokens at corresponding positions that match
- **BLEU Score**: N-gram overlap (0-100)
- **ROUGE-L**: Longest common subsequence F-measure
- **Edit Distance (Levenshtein)**: Character-level minimum edit operations
- **Semantic Similarity**: Cosine similarity of sentence embeddings
- **First Divergence Token**: Position where outputs first differ

## Output

The project generates:

- `results/raw/*.json`, Per-config inference outputs
- `results/reports/metrics_summary.csv`, Cross-config comparison table
- `results/reports/per_prompt_comparison.csv`, Per-prompt breakdown
- `results/reports/heatmap_*.png`, Visual comparison heatmaps
- `results/reports/full_report.md`, Comprehensive analysis report

## Project Structure

```
CSE5429/
├── configs/              # Experiment configurations
│   └── experiment_config.py
├── src/                  # Core modules
│   ├── prompts.py        # 20 standardized test prompts
│   ├── runner.py         # vLLM inference runner
│   └── utils.py          # JSON I/O, logging, GPU cleanup
├── evaluation/           # Comparison and reporting
│   ├── metrics.py        # BLEU, ROUGE, edit distance, etc.
│   ├── comparison.py     # Pairwise config comparison
│   └── report.py         # Report generation
├── scripts/              # Runner scripts
│   ├── run_experiment.py # Main CLI orchestrator
│   └── run_all.sh        # Shell wrapper
├── tests/                # Unit tests
├── results/              # Output (gitignored)
└── requirements.txt
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_metrics.py -v
```
