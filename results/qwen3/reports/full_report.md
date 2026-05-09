# LLM Inference Configuration Comparison Report

## Executive Summary

Optimization parallel inference configurations **do** change LLM outputs. 2 configuration(s) produced different outputs compared to the baseline: c3_cuda_graphs, c5_batch_processing.
 6 configuration(s) remained identical: c1_chunked_prefill_small, c2_chunked_prefill_large, c4_prefix_caching, c6_low_memory, c8_cpu_swap, c9_determinism.

## Configuration Summary

| Config | Display Name | Status |
|--------|-------------|--------|
| c0_baseline | C0: Baseline (No Optimizations) | ran |
| c1_chunked_prefill_small | C1: Chunked Prefill (512 tokens) | ran |
| c2_chunked_prefill_large | C2: Chunked Prefill (2048 tokens) | ran |
| c3_cuda_graphs | C3: CUDA Graphs Enabled | ran |
| c4_prefix_caching | C4: Prefix Caching Enabled | ran |
| c5_batch_processing | C5: Batch Processing | ran |
| c6_low_memory | C6: Low GPU Memory (40%) | ran |
| c8_cpu_swap | C8: CPU Offload (4 GB) | ran |
| c9_determinism | C9: Determinism Check (3 runs) | ran |

## Determinism Verification

C9 (determinism check): Baseline output is perfectly repeatable. All 20 prompt(s) produced exact matches across multiple runs.

## Cross-Configuration Comparison

| Config | Exact Match | BLEU | Edit Distance |
|--------|------------|------|---------------|
| c1_chunked_prefill_small | 1.0000 | 100.0000 | 0.0000 |
| c2_chunked_prefill_large | 1.0000 | 100.0000 | 0.0000 |
| c3_cuda_graphs | 0.4500 | 73.4504 | 124.1500 |
| c4_prefix_caching | 1.0000 | 100.0000 | 0.0000 |
| c5_batch_processing | 0.3000 | 73.3200 | 142.2000 |
| c6_low_memory | 1.0000 | 100.0000 | 0.0000 |
| c8_cpu_swap | 1.0000 | 100.0000 | 0.0000 |
| c9_determinism | 1.0000 | 100.0000 | 0.0000 |

## Key Findings

- **c1_chunked_prefill_small** vs c0_baseline: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c2_chunked_prefill_large** vs c0_baseline: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c3_cuda_graphs** vs c0_baseline: Outputs differ (exact_match=0.4500, BLEU=73.4504, edit_distance=124.1500).
- **c4_prefix_caching** vs c0_baseline: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c5_batch_processing** vs c0_baseline: Outputs differ (exact_match=0.3000, BLEU=73.3200, edit_distance=142.2000).
- **c6_low_memory** vs c0_baseline: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c8_cpu_swap** vs c0_baseline: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c9_determinism** vs c0_baseline: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c2_chunked_prefill_large** vs c1_chunked_prefill_small: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c3_cuda_graphs** vs c1_chunked_prefill_small: Outputs differ (exact_match=0.4500, BLEU=73.4504, edit_distance=124.1500).
- **c4_prefix_caching** vs c1_chunked_prefill_small: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c5_batch_processing** vs c1_chunked_prefill_small: Outputs differ (exact_match=0.3000, BLEU=73.3200, edit_distance=142.2000).
- **c6_low_memory** vs c1_chunked_prefill_small: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c8_cpu_swap** vs c1_chunked_prefill_small: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c9_determinism** vs c1_chunked_prefill_small: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c3_cuda_graphs** vs c2_chunked_prefill_large: Outputs differ (exact_match=0.4500, BLEU=73.4504, edit_distance=124.1500).
- **c4_prefix_caching** vs c2_chunked_prefill_large: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c5_batch_processing** vs c2_chunked_prefill_large: Outputs differ (exact_match=0.3000, BLEU=73.3200, edit_distance=142.2000).
- **c6_low_memory** vs c2_chunked_prefill_large: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c8_cpu_swap** vs c2_chunked_prefill_large: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c9_determinism** vs c2_chunked_prefill_large: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c4_prefix_caching** vs c3_cuda_graphs: Outputs differ (exact_match=0.4500, BLEU=73.4489, edit_distance=124.1500).
- **c5_batch_processing** vs c3_cuda_graphs: Outputs differ (exact_match=0.3500, BLEU=65.9003, edit_distance=166.2500).
- **c6_low_memory** vs c3_cuda_graphs: Outputs differ (exact_match=0.4500, BLEU=73.4489, edit_distance=124.1500).
- **c8_cpu_swap** vs c3_cuda_graphs: Outputs differ (exact_match=0.4500, BLEU=73.4489, edit_distance=124.1500).
- **c9_determinism** vs c3_cuda_graphs: Outputs differ (exact_match=0.4500, BLEU=73.4489, edit_distance=124.1500).
- **c5_batch_processing** vs c4_prefix_caching: Outputs differ (exact_match=0.3000, BLEU=73.3200, edit_distance=142.2000).
- **c6_low_memory** vs c4_prefix_caching: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c8_cpu_swap** vs c4_prefix_caching: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c9_determinism** vs c4_prefix_caching: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c6_low_memory** vs c5_batch_processing: Outputs differ (exact_match=0.3000, BLEU=73.3158, edit_distance=142.2000).
- **c8_cpu_swap** vs c5_batch_processing: Outputs differ (exact_match=0.3000, BLEU=73.3158, edit_distance=142.2000).
- **c9_determinism** vs c5_batch_processing: Outputs differ (exact_match=0.3000, BLEU=73.3158, edit_distance=142.2000).
- **c8_cpu_swap** vs c6_low_memory: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c9_determinism** vs c6_low_memory: Outputs are identical (exact_match=1.0, BLEU=100.0).
- **c9_determinism** vs c8_cpu_swap: Outputs are identical (exact_match=1.0, BLEU=100.0).

## Detailed Metrics

| Config A | Config B | Prompts | Exact Match | Token Match | BLEU | ROUGE-L | Edit Dist | Edit Dist (norm) | First Div |
|----------|----------|---------|------------|------------|------|--------|-----------|------------------|-----------|
| c0_baseline | c0_baseline | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c0_baseline | c1_chunked_prefill_small | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c0_baseline | c2_chunked_prefill_large | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c0_baseline | c3_cuda_graphs | 20 | 0.4500 | 0.6852 | 73.4504 | 0.7851 | 124.1500 | 0.2086 | 15.9500 |
| c0_baseline | c4_prefix_caching | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c0_baseline | c5_batch_processing | 20 | 0.3000 | 0.5844 | 73.3200 | 0.7828 | 142.2000 | 0.2377 | 22.1000 |
| c0_baseline | c6_low_memory | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c0_baseline | c8_cpu_swap | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c0_baseline | c9_determinism | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c1_chunked_prefill_small | c1_chunked_prefill_small | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c1_chunked_prefill_small | c2_chunked_prefill_large | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c1_chunked_prefill_small | c3_cuda_graphs | 20 | 0.4500 | 0.6852 | 73.4504 | 0.7851 | 124.1500 | 0.2086 | 15.9500 |
| c1_chunked_prefill_small | c4_prefix_caching | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c1_chunked_prefill_small | c5_batch_processing | 20 | 0.3000 | 0.5844 | 73.3200 | 0.7828 | 142.2000 | 0.2377 | 22.1000 |
| c1_chunked_prefill_small | c6_low_memory | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c1_chunked_prefill_small | c8_cpu_swap | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c1_chunked_prefill_small | c9_determinism | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c2_chunked_prefill_large | c2_chunked_prefill_large | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c2_chunked_prefill_large | c3_cuda_graphs | 20 | 0.4500 | 0.6852 | 73.4504 | 0.7851 | 124.1500 | 0.2086 | 15.9500 |
| c2_chunked_prefill_large | c4_prefix_caching | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c2_chunked_prefill_large | c5_batch_processing | 20 | 0.3000 | 0.5844 | 73.3200 | 0.7828 | 142.2000 | 0.2377 | 22.1000 |
| c2_chunked_prefill_large | c6_low_memory | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c2_chunked_prefill_large | c8_cpu_swap | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c2_chunked_prefill_large | c9_determinism | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c3_cuda_graphs | c3_cuda_graphs | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c3_cuda_graphs | c4_prefix_caching | 20 | 0.4500 | 0.6852 | 73.4489 | 0.7851 | 124.1500 | 0.2086 | 15.9500 |
| c3_cuda_graphs | c5_batch_processing | 20 | 0.3500 | 0.5602 | 65.9003 | 0.7244 | 166.2500 | 0.2809 | 15.1000 |
| c3_cuda_graphs | c6_low_memory | 20 | 0.4500 | 0.6852 | 73.4489 | 0.7851 | 124.1500 | 0.2086 | 15.9500 |
| c3_cuda_graphs | c8_cpu_swap | 20 | 0.4500 | 0.6852 | 73.4489 | 0.7851 | 124.1500 | 0.2086 | 15.9500 |
| c3_cuda_graphs | c9_determinism | 20 | 0.4500 | 0.6852 | 73.4489 | 0.7851 | 124.1500 | 0.2086 | 15.9500 |
| c4_prefix_caching | c4_prefix_caching | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c4_prefix_caching | c5_batch_processing | 20 | 0.3000 | 0.5844 | 73.3200 | 0.7828 | 142.2000 | 0.2377 | 22.1000 |
| c4_prefix_caching | c6_low_memory | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c4_prefix_caching | c8_cpu_swap | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c4_prefix_caching | c9_determinism | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c5_batch_processing | c5_batch_processing | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c5_batch_processing | c6_low_memory | 20 | 0.3000 | 0.5844 | 73.3158 | 0.7828 | 142.2000 | 0.2377 | 22.1000 |
| c5_batch_processing | c8_cpu_swap | 20 | 0.3000 | 0.5844 | 73.3158 | 0.7828 | 142.2000 | 0.2377 | 22.1000 |
| c5_batch_processing | c9_determinism | 20 | 0.3000 | 0.5844 | 73.3158 | 0.7828 | 142.2000 | 0.2377 | 22.1000 |
| c6_low_memory | c6_low_memory | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c6_low_memory | c8_cpu_swap | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c6_low_memory | c9_determinism | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c8_cpu_swap | c8_cpu_swap | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c8_cpu_swap | c9_determinism | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
| c9_determinism | c9_determinism | 20 | 1.0000 | 1.0000 | 100.0000 | 1.0000 | 0.0000 | 0.0000 | -1.0000 |
