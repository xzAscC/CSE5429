"""Subprocess script for running a single vLLM config in isolation."""
import json
import sys
import os

sys.path.insert(0, os.getcwd())

[config_name, display_name, llm_kwargs_json, prompts_json,
 batch_mode, num_runs, max_tokens, result_path] = sys.argv[1:]

llm_kwargs = json.loads(llm_kwargs_json)
prompts_data = json.loads(prompts_json)
batch_mode = batch_mode == "True"
num_runs = int(num_runs)
max_tokens = int(max_tokens)

import vllm

result = {
    "config_name": config_name,
    "display_name": display_name,
    "outputs": [],
    "error": None,
    "skipped": False,
    "skip_reason": None,
}

try:
    sampling_params = vllm.SamplingParams(
        temperature=0, max_tokens=max_tokens, seed=42
    )

    for run_idx in range(num_runs):
        llm = vllm.LLM(**llm_kwargs)
        try:
            if batch_mode:
                texts = [p["text"] for p in prompts_data]
                completions = llm.generate(texts, sampling_params)
                for pd, comp in zip(prompts_data, completions):
                    text = comp.outputs[0].text
                    token_ids = list(comp.outputs[0].token_ids)
                    result["outputs"].append({
                        "prompt_id": pd["id"],
                        "prompt_text": pd["text"],
                        "output_text": text,
                        "token_ids": token_ids,
                        "num_generated_tokens": len(token_ids),
                        "config_name": config_name,
                        "run_index": run_idx,
                    })
            else:
                for pd in prompts_data:
                    completions = llm.generate([pd["text"]], sampling_params)
                    text = completions[0].outputs[0].text
                    token_ids = list(completions[0].outputs[0].token_ids)
                    result["outputs"].append({
                        "prompt_id": pd["id"],
                        "prompt_text": pd["text"],
                        "output_text": text,
                        "token_ids": token_ids,
                        "num_generated_tokens": len(token_ids),
                        "config_name": config_name,
                        "run_index": run_idx,
                    })
        finally:
            del llm
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

except Exception as exc:
    result["error"] = str(exc)

with open(result_path, "w") as f:
    json.dump(result, f)
