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
    """
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


# --------------------------------------------------------------------------- #
# Base prompt: instructions + STRICT contract only.
# The code skeleton (with the trusted primitives objective() and
# best_insertion_position() and the self-healing return) is NOT embedded here;
# it is supplied by ./template.py, which the handler loads. The prompt below
# references those primitives, so template.py MUST define them.
# --------------------------------------------------------------------------- #
BASE_PROMPT = (
    "You are an expert metaheuristic designer and algorithm engineer specializing in "
    "combinatorial optimization.\n\n"

    "### Task\n"
    "Implement a highly effective, innovative metaheuristic for the Linear Ordering Problem (LOP) "
    "by filling ONLY the TODO block of the provided code skeleton.\n\n"

    "### Problem Definition\n"
    "Given a dense, non-symmetric n x n matrix C, find a permutation pi of {0,...,n-1} that maximizes\n"
    "  f(pi) = sum_{i<j} C[pi(i), pi(j)]\n"
    "Key structural fact: moving element e immediately past element x flips exactly that one pair, so "
    "f changes by C[x, e] - C[e, x]. A cumulative sum over the other elements therefore gives e's best "
    "insertion position in O(n). Exploit this; it is already implemented for you in the skeleton.\n\n"

    "### Instance & Performance Target\n"
    "- Scale: n = 100. Time budget: return within ~15s wall-clock; track time.time() and stop early.\n"
    "- Aim to complete thousands of neighborhood sweeps, not a handful. Near-optimal quality.\n\n"

    "### Use the provided primitives (do NOT reimplement evaluation)\n"
    "The skeleton already defines, and you MUST build on:\n"
    "  - objective(perm): exact vectorized O(n^2) objective. Use ONLY for accept/compare decisions, "
    "never inside an inner neighborhood loop.\n"
    "  - best_insertion_position(rest, e): exact O(n) best gap + contribution for inserting element e "
    "into partial permutation `rest`. This is your fast move evaluator.\n"
    "Do NOT write your own delta/incremental evaluator and do NOT recompute the full objective inside "
    "neighborhood scans. Reinventing delta-evaluation is the dominant source of index bugs.\n\n"

    "### Innovation & Constraints\n"
    "- No textbook Tabu Search / Simulated Annealing / vanilla GA. Add a genuine structural idea on TOP "
    "of the insertion neighborhood (ILS/VNS kicks, ALNS destroy-repair, GRASP + path-relinking).\n"
    "- INDEX DISCIPLINE: keep `positions` (where in the list) and `element labels` (which node) strictly "
    "separate. Never index C with a position, nor the perm list with a label. Any segment operation "
    "(reversal, block move) MUST leave the result a permutation of range(n).\n\n"

    "### Output Contract (STRICT, machine-checked)\n"
    "Rejected unless ALL hold: (1) main returns a plain Python list, NOT a NumPy array; (2) length "
    "exactly n; (3) a permutation of range(n) -- no duplicates, no missing indices; (4) returns within "
    "budget and raises no exception.\n"
    "AVOID these known-rejected outputs: returning np.argsort(...) directly (a NumPy array); a "
    "partial/duplicated permutation from a broken construction or repair; calling a helper you never "
    "defined.\n\n"

    "### Code rules\n"
    "- Fill ONLY the TODO of the provided skeleton. Keep the def main signature, the import markers, the "
    "TRUSTED PRIMITIVES, and the SELF-HEALING RETURN exactly as given -- do not modify or delete them.\n"
    "- Define every one of your own helpers INSIDE main, and define each BEFORE it is called.\n"
    "- `solution` is pre-initialized to a valid permutation; keep it valid at all times so an early "
    "time-out still returns something legal.\n"
    "- No prose, no printing. Respond with EXACTLY ONE python code block: the completed skeleton, nothing else.\n"
)


# --------------------------------------------------------------------------- #
# Feedback helpers
# --------------------------------------------------------------------------- #
def _gen_fitness_history(records, instance_num):
    """Best fitness per generation (None for generations with no valid solution)."""
    hist = []
    for start in range(0, len(records), instance_num):
        block = records[start:start + instance_num]
        fits = [r["fitness"] for r in block if r["success"] and r["fitness"] is not None]
        hist.append(max(fits) if fits else None)
    return hist


def _last_error(records, instance_num):
    last = records[-instance_num:]
    for r in last:
        if not r["success"]:
            return r.get("error_type") or "unknown", (r.get("error") or "")
    return None, ""


def decide_mode(records, instance_num, last_gen_valid, have_valid_base, patience=4):
    """Return (mode_name, directive) controlling how aggressively the next step changes the code."""
    if not have_valid_base:
        return "COLD_START", (
            "MODE = COLD_START. No valid implementation exists yet. Do NOT innovate. Produce the "
            "SIMPLEST correct algorithm: repeated best-insertion local search using "
            "best_insertion_position() (sweep every element to its best gap until no improvement), "
            "restarting from random orders while time remains. Priority #1 is a VALID return."
        )

    if not last_gen_valid:
        et, err = _last_error(records, instance_num)
        snippet = (": " + err[:160]) if err else ""
        return "REPAIR", (
            f"MODE = REPAIR. Your most recent attempt FAILED ({et}{snippet}). Start from the "
            "known-valid base below and make the MINIMAL change needed to fix the fault. Add NO new "
            "ideas this round. Do not touch the trusted primitives or the self-healing return."
        )

    hist = _gen_fitness_history(records, instance_num)
    valid_fits = [h for h in hist if h is not None]
    if len(valid_fits) <= patience:
        return "INTENSIFY", (
            "MODE = INTENSIFY. Keep the working core unchanged. Make a small, targeted refinement "
            "(tune acceptance, deepen the neighborhood, or improve the perturbation). Do NOT rewrite "
            "the core or the primitives."
        )

    best_before = max(valid_fits[:-patience])
    recent_best = max(valid_fits[-patience:])
    if recent_best <= best_before * (1.0 + 1e-4):
        return "EXPLORE", (
            f"MODE = EXPLORE. Fitness has plateaued over the last {patience} valid generations. "
            "Introduce exactly ONE bold structural change (a new operator, neighborhood, or "
            "metaheuristic layer) on top of the base below. If it underperforms we revert to this "
            "base, so you risk nothing -- but KEEP the trusted primitives and the return contract intact."
        )
    return "INTENSIFY", (
        "MODE = INTENSIFY. Fitness is still improving. Keep the working core and refine it with a "
        "small, targeted change. Do NOT rewrite the core or the primitives."
    )


def generate_feedback_hints(records, instance_num):
    """Translate the last generation's real error_types into architectural guidance."""
    last = records[-instance_num:]
    failures = [r for r in last if not r["success"]]

    if not failures:
        return (
            "- The current strategy is valid and stable. Improve QUALITY only: deepen the insertion "
            "neighborhood, add path-relinking or an oscillating acceptance rule, and spend the full "
            "budget intensifying around the incumbent. Use the provided primitives for all evaluation."
        )

    types = {r.get("error_type") or "unknown" for r in failures}
    errs = " ".join(str(r.get("error")) for r in failures if r.get("error"))
    hints = [
        "- Use objective() and best_insertion_position() for ALL evaluation. Do NOT hand-write a delta "
        "evaluator or recompute fitness in neighborhood loops -- that is the main source of these faults."
    ]

    if types & {"compile", "no_result", "empty"}:
        hints.append(
            "- CRITICAL: output did not compile or was incomplete. Return EXACTLY ONE complete "
            "```python``` block defining main(instance). No prose, no truncation."
        )
    if "invalid_solution" in types:
        hints.append(
            "- CRITICAL: main() returned something that is not a permutation of range(n) (most often a "
            "NumPy array, or a duplicated/partial list). Keep `solution` a plain Python list of n "
            "distinct ints and do NOT remove the self-healing return."
        )
    if "timeout" in types:
        hints.append(
            "- CRITICAL: TIMED OUT. Respect MAX_TIME via time.time(); rely on the O(n) "
            "best_insertion_position primitive instead of any O(n^2) recompute inside loops."
        )
    if "runtime" in types:
        sub = []
        if "IndexError" in errs or "KeyError" in errs:
            sub.append(
                "indexing errors -- keep positions and element labels separate; never index C with a "
                "position; ensure every segment/move keeps a full permutation"
            )
        if "NameError" in errs:
            sub.append("a helper was used before being defined -- define every helper inside main first")
        if "MemoryError" in errs or "Memory" in errs:
            sub.append("excessive memory -- keep data structures flat, avoid large populations/trees")
        detail = ("; specifically: " + "; ".join(sub)) if sub else ""
        hints.append(
            f"- CRITICAL: an exception was raised at runtime{detail}. Add guards and validate "
            "intermediate permutations."
        )

    return "\n".join(hints)


def summarize_results(records, instance_num):
    """Compact summary of the last generation's results."""
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
# Prompt construction (state-aware: cold-start / repair / intensify / explore)
# Full previous solution is injected uncapped (H100 has ample context budget).
# --------------------------------------------------------------------------- #
def get_prompt(i, base_prompt, records, instance_num, code, last_valid_code=None):
    if i == 0:
        return base_prompt

    summary = summarize_results(records, instance_num)
    feedback_hints = generate_feedback_hints(records, instance_num)
    last_gen_valid = any(r["success"] for r in records[-instance_num:])
    have_valid_base = last_valid_code is not None

    mode, mode_directive = decide_mode(
        records, instance_num, last_gen_valid, have_valid_base
    )

    if have_valid_base:
        shown_code = last_valid_code
        shown_label = "Best Known-Valid Implementation"
    else:
        shown_code = code or ""
        shown_label = "Your Previous (Invalid) Attempt"

    prompt = (
        f"{base_prompt}\n\n"
        f"=== ITERATION {i} FEEDBACK LOOP ===\n"
        f"You are inside an iterative optimization loop; your previous code was executed and analyzed.\n\n"
        f"**Performance on {instance_num} benchmark instance(s):**\n{summary}\n\n"
        f"**{mode_directive}**\n\n"
        f"**Strategic guidance:**\n{feedback_hints}\n\n"
        f"**{shown_label}:**\n```python\n{shown_code}\n```\n\n"
        f"Produce the next implementation following the MODE above. Obey the strict output contract: "
        f"EXACTLY ONE python code block defining main(instance) that returns a permutation of range(n); "
        f"keep the trusted primitives and self-healing return unchanged; no prose."
    )
    return prompt


# --------------------------------------------------------------------------- #
# Validation helper (mirrors the contract)
# --------------------------------------------------------------------------- #
def is_valid_solution(solution, success, n):
    return (
        success
        and isinstance(solution, list)
        and len(solution) == n
        and all(isinstance(x, (int, np.integer)) for x in solution)
        and len(set(solution)) == n
        and set(solution) == set(range(n))
    )


# --------------------------------------------------------------------------- #
# Main optimization loop
# --------------------------------------------------------------------------- #
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
    last_valid_code = None  # no seed; until first valid solution, decide_mode uses COLD_START

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
        # The skeleton (with primitives + self-healing return) comes from template.py.
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
            is_valid = is_valid_solution(result.solution, result.success, n)
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
