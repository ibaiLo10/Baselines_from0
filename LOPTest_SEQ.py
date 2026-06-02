import re
import os

import numpy as np
import pandas as pd
import torch

import LLMhandling
import LOPbasics


# --------------------------------------------------------------------------- #
# Code extraction
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the Python code out of an LLM response.

    Handles three cases:
      - prose + one or more fenced ```python blocks  -> returns the largest block
      - a single fenced block                         -> returns its contents
      - pure code with no fences                      -> returns the text as-is

    Applied right after get_response so that everything downstream
    (saved file, tester, feedback loop) works on clean code, never prose.
    """
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()  # the longest fenced block
    return text.strip()


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def cap_code(code: str, max_chars: int = 16000) -> str:
    """Cap the previous solution re-injected into the prompt.

    Keeps inline comments (useful context for the next iteration); only trims
    pathologically large files so the prompt does not blow up.
    """
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n# ... [truncated for brevity]"


def get_prompt(i, base_prompt, records, instance_num, code, last_valid_code=None):
    if i == 0:
        return base_prompt

    summary = summarize_results(records, instance_num)
    feedback_hints = generate_feedback_hints(records, instance_num)

    last_gen_valid = any(r["success"] for r in records[-instance_num:])

    # Decide which code to show as the base to improve.
    if last_valid_code:
        base_for_prompt = last_valid_code
        if last_gen_valid:
            shown_label = "Your Best Working Implementation So Far"
            task = (
                "Improve the implementation below. Keep its working algorithmic core, but make a "
                "major innovative modification to push past the current fitness plateau."
            )
        else:
            shown_label = "Last Valid Implementation"
            task = (
                "The implementation below is the last one that produced a VALID solution. Your most "
                "recent attempt FAILED (see guidance above). Start from this valid base, fix the issue, "
                "and improve it."
            )
    else:
        base_for_prompt = code or ""
        shown_label = "Your Previous (Invalid) Attempt"
        task = (
            "Your previous attempt did NOT produce a valid solution (see guidance above). Produce a "
            "complete, correct, executable implementation that satisfies the output contract."
        )

    clean_code = cap_code(base_for_prompt)

    prompt = (
        f"{base_prompt}\n\n"
        f"=== ITERATION {i} FEEDBACK LOOP ===\n"
        f"You are inside an iterative optimization loop. Your previous implementation was executed and analyzed.\n\n"
        f"**Performance Metrics on {instance_num} Benchmark Instance(s):**\n"
        f"{summary}\n\n"
        f"**Critical Strategic Guidance:**\n"
        f"{feedback_hints}\n\n"
        f"**{shown_label}:**\n"
        f"```python\n"
        f"{clean_code}\n"
        f"```\n\n"
        f"Task: {task}\n"
        f"Respect the strict output contract: respond with EXACTLY ONE python code block defining "
        f"main(instance) that returns a permutation of range(n). No prose before or after the block."
    )
    return prompt


def generate_feedback_hints(records, instance_num):
    """Translate the last generation's real error_types into architectural guidance."""
    last = records[-instance_num:]
    failures = [r for r in last if not r["success"]]

    if not failures:
        return (
            "- The current strategy is valid and stable. Focus strictly on solution QUALITY: deepen "
            "the neighborhood structure, add path-relinking or an oscillating acceptance criterion, and "
            "spend the full time budget intensifying around the best solution found."
        )

    types = {r.get("error_type") or "unknown" for r in failures}
    errs = " ".join(str(r.get("error")) for r in failures if r.get("error"))
    hints = []

    if types & {"compile", "no_result", "empty"}:
        hints.append(
            "- CRITICAL: your previous output did not compile or was incomplete/empty. Return EXACTLY "
            "ONE complete ```python``` block defining `main(instance)`. No prose, no truncation."
        )
    if "invalid_solution" in types:
        hints.append(
            "- CRITICAL: main() did not return a valid solution. It MUST return a Python list of n "
            "distinct integers forming a permutation of range(n): no duplicates, no missing indices, "
            "length exactly n. Do not return a NumPy array."
        )
    if "timeout" in types:
        hints.append(
            "- CRITICAL: your algorithm TIMED OUT. Respect the time budget using time.time(), and use "
            "incremental delta-evaluation so swaps/inserts cost O(1)/O(n) instead of recomputing fitness "
            "in O(n^2). Prune the search space."
        )
    if "runtime" in types:
        sub = []
        if "IndexError" in errs or "KeyError" in errs:
            sub.append(
                "indexing errors (check matrix bounds 0..n-1 and make sure permutation moves neither "
                "duplicate nor drop elements)"
            )
        if "MemoryError" in errs or "Memory" in errs:
            sub.append(
                "excessive memory (avoid huge explicit populations or deep search trees; keep data "
                "structures flat)"
            )
        detail = ("; specifically: " + ", ".join(sub)) if sub else ""
        hints.append(
            f"- CRITICAL: your code raised an exception at runtime{detail}. Add guards for edge cases "
            f"and validate intermediate permutations."
        )

    return "\n".join(hints)


def summarize_results(records, instance_num):
    """Compact summary of the last generation's results (avoids embedding the
    unbounded raw records list into the prompt)."""
    last = records[-instance_num:]
    successes = [r for r in last if r["success"]]
    failures = [r for r in last if not r["success"]]

    if successes:
        fitnesses = [r["fitness"] for r in successes]
        mean_fit = round(float(np.mean(fitnesses)), 1)
        best_fit = round(float(np.max(fitnesses)), 1)
        worst_fit = round(float(np.min(fitnesses)), 1)
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
        f"Fitness -- best: {best_fit}, mean: {mean_fit}, worst: {worst_fit} | "
        f"Error breakdown: {error_counts}"
    )


# --------------------------------------------------------------------------- #
# Base prompt: CODE ONLY + explicit I/O contract that mirrors the validator
# --------------------------------------------------------------------------- #
BASE_PROMPT = (
    "You are an expert metaheuristic designer and algorithm engineer specializing in combinatorial optimization.\n\n"
    "### Task\n"
    "Implement a highly effective, innovative metaheuristic algorithm to solve the Linear Ordering Problem (LOP).\n\n"
    "### Problem Definition\n"
    "Given a dense, non-symmetric n x n matrix C, find a permutation pi of {0, 1, ..., n-1} that maximizes "
    "the sum of the upper-triangular elements of the reordered matrix:\n"
    "  f(pi) = sum_{i=0}^{n-2} sum_{j=i+1}^{n-1} C[pi(i), pi(j)]\n\n"
    "### Instance Characteristics & Performance Target\n"
    "- Scale: n = 100.\n"
    "- Time budget: main(instance) MUST return within ~15 seconds of wall-clock time. Track time with "
    "time.time() and stop the search before the budget is exhausted.\n"
    "- Quality: aim for near-optimal solutions, minimizing the gap to known LOP benchmarks.\n\n"
    "### Innovation & Constraints\n"
    "- Do not provide a textbook Tabu Search, Simulated Annealing, or basic Genetic Algorithm.\n"
    "- Introduce an innovative structural idea. Consider frameworks such as:\n"
    "  - Adaptive Large Neighborhood Search (ALNS) with novel destroy/repair operators for LOP.\n"
    "  - Variable Neighborhood Search (VNS) with block-insertion or sub-sequence inversion.\n"
    "  - GRASP combined with a path-relinking strategy.\n"
    "- Use incremental delta-evaluation so each neighborhood move costs O(1)/O(n), not a full O(n^2) recompute.\n"
    "- Balance exploration (diversification) and exploitation (intensification).\n"
    "- Use NumPy and keep data structures flat and efficient.\n\n"
    "### Output Contract (STRICT)\n"
    "- A Python code skeleton is provided at the END of this prompt. Fill in ONLY its TODO "
    "section; keep the `def main(instance):` signature, the import markers, and the final return.\n"
    "- `instance` is a NumPy array of shape (n, n). `main` MUST return a Python list of n distinct "
    "integers: a permutation of range(n). Not a NumPy array.\n"
    "- Do NOT print. Do NOT write any explanation or prose before or after the code. "
    "Inline comments are fine.\n"
    "- Respond with EXACTLY ONE Python code block containing the completed skeleton and nothing else.\n"
    "# complete, self-contained, executable implementation\n"
    "def main(instance):\n"
    "    ...\n"
    "    return permutation  # list[int], a permutation of range(len(instance))\n"
    "```\n"
)


if __name__ == "__main__":
    np.random.seed(42)
    INSTANCE_NUM = 1
    NUM_GENERATIONS = 100
    TIMEOUT_SECONDS = 7200  # safety margin above the ~15s budget stated in the prompt

    instances = [np.random.randint(0, 100, (100, 100)) for _ in range(INSTANCE_NUM)]
    model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    model_args = {
        "temperature": 0.8,
        "max_new_tokens": 16000,
    }

    # Load model once — no server needed
    handler = LLMhandling.LLMHandler(mode="hf", model_name=model, model_args=model_args)

    records = []
    os.makedirs("algorithms", exist_ok=True)
    code = None
    last_valid_code = None

    for i in range(NUM_GENERATIONS):
        algorithm_id = f"algorithm_{i}"

        # --- Build prompt ---------------------------------------------------
        if i == 0:
            prompt = BASE_PROMPT
        else:
            prompt = get_prompt(
                i=i,
                base_prompt=BASE_PROMPT,
                records=records,
                instance_num=INSTANCE_NUM,
                code=code,
                last_valid_code=last_valid_code,
            )

        # --- Generate and extract clean code -------------------------------
        raw = handler.get_response(template_path="./template.py", prompt=prompt)
        code = extract_code(raw)  # clean code from here on (no prose, no fences)

        with open(f"algorithms/{algorithm_id}.py", "w") as f:
            f.write(code)

        # --- Evaluate -------------------------------------------------------
        gen_valid = False
        for j, instance in enumerate(instances):
            tester = LLMhandling.CodeTester(instance=instance, timeout=TIMEOUT_SECONDS)
            result = tester.test(code)

            n = instance.shape[0]
            is_valid = (
                result.success
                and isinstance(result.solution, list)
                and len(result.solution) == n
                and all(isinstance(x, (int, np.integer)) for x in result.solution)
                and len(set(result.solution)) == n
                and set(result.solution) == set(range(n))
            )
            if is_valid:
                gen_valid = True

            fitness = LOPbasics.fitness_function(result.solution, instance) if is_valid else None

            records.append({
                "algorithm_id": algorithm_id,
                "instance_id": j,
                "fitness": fitness,
                "success": is_valid,
                "error_type": result.error_type if not result.success else (
                    None if is_valid else "invalid_solution"
                ),
                "error": result.error if not result.success else None,
            })

        # Keep the last code that produced a valid solution so a failed
        # generation never poisons the next prompt.
        if gen_valid:
            last_valid_code = code

        # --- Free GPU memory between LLM calls ------------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv("results.csv", index=False)
