import numpy as np
import LLMhandling
import pandas as pd
import os
import torch

import LOPbasics


def summarize_results(records, instance_num):
    """Produce a compact summary of the last generation's results instead of
    embedding the raw records list (which grows unboundedly and bloats the prompt)."""
    last = records[-instance_num:]
    successes = [r for r in last if r["success"]]
    failures  = [r for r in last if not r["success"]]

    if successes:
        fitnesses   = [r["fitness"] for r in successes]
        mean_fit    = round(float(np.mean(fitnesses)), 1)
        best_fit    = round(float(np.max(fitnesses)), 1)
        worst_fit   = round(float(np.min(fitnesses)), 1)
    else:
        mean_fit = best_fit = worst_fit = None

    error_counts: dict = {}
    for r in failures:
        et = r["error_type"] or "unknown"
        error_counts[et] = error_counts.get(et, 0) + 1

    return (
        f"Instances evaluated: {len(last)} | "
        f"Successful: {len(successes)} | "
        f"Failed: {len(failures)} | "
        f"Fitness — best: {best_fit}, mean: {mean_fit}, worst: {worst_fit} | "
        f"Error breakdown: {error_counts}"
    )


def truncate_code(code: str, max_chars: int = 3000) -> str:
    """Keep only the first max_chars characters of generated code to cap
    the token count when feeding it back into the next prompt."""
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n# ... [truncated for brevity]"


BASE_PROMPT = """
You are an expert optimization algorithm designer. Your task is to implement an algorithm
for the Linear Ordering Problem (LOP). Given an n×n matrix, the goal is to find a permutation
of rows and columns that maximizes the sum of the upper triangle of the reordered matrix.
The matrix rows and columns are indexed 0 to n-1. The solution must be a permutation: a list
containing each integer from 0 to n-1 exactly once. The algorithm should be computationally
efficient and practical for instances of size n=100. Aim for the highest solution quality possible.
"""


if __name__ == "__main__":
    np.random.seed(42)
    INSTANCE_NUM = 1
    NUM_GENERATIONS = 100
    instances = [np.random.randint(0, 100, (100, 100)) for _ in range(INSTANCE_NUM)]
    model = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
    model_args = {
        "temperature": 0.8,
        "max_new_tokens": 2500
    }

    # Load model once — no server needed
    handler = LLMhandling.LLMHandler(mode='hf', model_name=model, model_args=model_args)

    records = []
    os.makedirs("algorithms", exist_ok=True)
    code = None

    for i in range(NUM_GENERATIONS):
        algorithm_id = f"algorithm_{i}"

        # --- Build prompt ---------------------------------------------------
        if i == 0:
            prompt = BASE_PROMPT
        else:
            # Use a compact summary instead of raw records + full code to keep
            # the prompt token count constant across generations.
            summary      = summarize_results(records, INSTANCE_NUM)
            short_code   = truncate_code(code, max_chars=3000)
            prompt = (
                BASE_PROMPT +
                f"\nThe previous algorithm achieved the following on {INSTANCE_NUM} instances:\n"
                f"{summary}\n\n"
                f"Here is the previous algorithm's code (study it and improve upon it, do not leave it there, remove what unnecessary):\n"
                f"{short_code}\n"
            )

        code = handler.get_response(template_path="./template.py", prompt=prompt)

        with open(f"algorithms/{algorithm_id}.py", "w") as f:
            f.write(code)

        # --- Evaluate -------------------------------------------------------
        for j, instance in enumerate(instances):
            tester = LLMhandling.CodeTester(instance=instance, timeout=300)
            result = tester.test(code)

            n = instance.shape[0]
            is_valid = (
                result.success and
                isinstance(result.solution, list) and
                len(result.solution) == n and
                all(isinstance(x, (int, np.integer)) for x in result.solution) and
                len(set(result.solution)) == n and
                set(result.solution) == set(range(n))
            )
            fitness = LOPbasics.fitness_function(result.solution, instance) if is_valid else None

            records.append({
                "algorithm_id": algorithm_id,
                "instance_id":  j,
                "fitness":      fitness,
                "success":      is_valid,
                "error_type":   result.error_type if not result.success else (
                                    None if is_valid else "invalid_solution"),
            })

        # --- Free GPU memory between LLM calls ------------------------------
        # Clears the allocator cache so fragmented blocks don't accumulate
        # over 100 generations and push usage over the VRAM limit.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv("results.csv", index=False)
