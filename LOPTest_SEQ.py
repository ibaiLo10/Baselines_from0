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
# The code skeleton (a dynamic seed plus a self-healing return) is NOT embedded
# here; it is supplied by ./template.py, which the handler loads. There are NO
# evaluation primitives in the skeleton: the model only has to return a
# permutation, and its fitness is scored externally by this loop (via
# LOPbasics.fitness_function).
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
    "Useful structural fact you can exploit: moving element e immediately past element x flips exactly "
    "that one pair, so f changes by C[x, e] - C[e, x]. A cumulative sum over the other elements "
    "therefore yields e's best insertion position in O(n) -- a strong basis for fast moves if you "
    "choose to implement one.\n\n"

    "### Evaluation is external -- you do NOT implement it\n"
    "You do NOT need to implement, call, or return any objective/fitness function: the quality of your "
    "permutation is scored OUTSIDE this code by the optimization loop. There are NO pre-provided "
    "helpers or primitives in the skeleton. Your single deliverable is a valid permutation (see the "
    "contract below). If your search benefits from comparing candidate solutions internally, you are "
    "free to implement that yourself -- but define every helper INSIDE main().\n\n"

    "### Instance & Performance Target\n"
    "- Scale: n = 100.\n"
    "- Time budget: 10 minutes is the HARD MAXIMUM allowed wall-clock, NOT a duration you must fill. "
    "Track time.time() and you MUST return before it elapses. You may keep improving the solution "
    "while time remains and it keeps helping; once the search has converged, return early -- there is "
    "no benefit to running out the clock, and never pad or sleep to reach the limit.\n"
    "- Aim for near-optimal quality: fast individual moves let you run many sweeps when they keep "
    "helping, but stop once they no longer do.\n\n"

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
    "dynamic seed, and the SELF-HEALING RETURN exactly as given -- do not modify or delete them.\n"
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
            "SIMPLEST correct algorithm: a repeated best-insertion local search (for each element, "
            "evaluate moving it to every gap and keep the best move; sweep until no improvement), "
            "restarting from random orders while time remains. Implement the move evaluation yourself. "
            "Priority #1 is a VALID permutation return."
        )

    if not last_gen_valid:
        et, err = _last_error(records, instance_num)
        snippet = (": " + err[:160]) if err else ""
        return "REPAIR", (
            f"MODE = REPAIR. Your most recent attempt FAILED ({et}{snippet}). Start from the "
            "known-valid base below and make the MINIMAL change needed to fix the fault. Add NO new "
            "ideas this round. Do not touch the dynamic seed or the self-healing return."
        )

    hist = _gen_fitness_history(records, instance_num)
    valid_fits = [h for h in hist if h is not None]
    if len(valid_fits) <= patience:
        return "INTENSIFY", (
            "MODE = INTENSIFY. Keep the working core unchanged. Make a small, targeted refinement "
            "(tune acceptance, deepen the neighborhood, or improve the perturbation). Do NOT rewrite "
            "the working core."
        )

    best_before = max(valid_fits[:-patience])
    recent_best = max(valid_fits[-patience:])
    if recent_best <= best_before * (1.0 + 1e-4):
        return "EXPLORE", (
            f"MODE = EXPLORE. Fitness has plateaued over the last {patience} valid generations. "
            "Introduce exactly ONE bold structural change (a new operator, neighborhood, or "
            "metaheuristic layer) on top of the base below. If it underperforms we revert to this "
            "base, so you risk nothing -- but KEEP the self-healing return and the output contract intact."
        )
    return "INTENSIFY", (
        "MODE = INTENSIFY. Fitness is still improving. Keep the working core and refine it with a "
        "small, targeted change. Do NOT rewrite the working core."
    )


def generate_feedback_hints(records, instance_num):
    """Translate the last generation's real error_types into architectural guidance."""
    last = records[-instance_num:]
    failures = [r for r in last if not r["success"]]

    if not failures:
        return (
            "- The current strategy is valid and stable. Improve QUALITY only: deepen the insertion "
            "neighborhood, add path-relinking or an oscillating acceptance rule, and intensify around "
            "the incumbent -- but stop once it stops improving (the time budget is a ceiling, not a "
            "quota). All evaluation is your own; nothing is pre-provided."
        )

    types = {r.get("error_type") or "unknown" for r in failures}
    errs = " ".join(str(r.get("error")) for r in failures if r.get("error"))
    hints = [
        "- Implement your own evaluation/move logic and reuse it consistently; a mismatch between the "
        "score you optimize and the permutation you actually return is a common source of these faults. "
        "Validate that every move keeps a full permutation of range(n)."
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
            "- CRITICAL: TIMED OUT. Respect the time budget via time.time() and return BEFORE it "
            "elapses (it is a hard ceiling). Use an O(n) incremental move evaluation you implement "
            "instead of recomputing the full O(n^2) objective inside inner loops."
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
# Reference-code selection for the feedback prompt
#
# We feed the model up to THREE prior versions, each accompanied by its
# per-instance fitness on the SAME benchmark instances:
#   1) PREVIOUS    -> the immediately preceding generation,
#   2) LAST VALID  -> the most recent generation that returned a valid solution,
#   3) BEST        -> the best generation so far (most valid instances, then
#                     highest mean fitness over the instances).
#
# For speed / context economy we NEVER show the same code twice: roles that map
# to the same generation (or to byte-identical code) are collapsed into a single
# block. Consequences of this rule:
#   - If the previous run was VALID it already IS the last-valid one, so PREVIOUS
#     and LAST VALID collapse (and BEST too if it is that same generation).
#   - All THREE distinct blocks appear only when the previous run FAILED (so it
#     differs from the last-valid one) AND the best so far is an older version
#     than the last-valid one (so best != last-valid).
# --------------------------------------------------------------------------- #
ROLE_LABELS = {
    "previous": "PREVIOUS attempt (immediately preceding generation)",
    "last_valid": "LAST VALID implementation",
    "best": "BEST-so-far implementation",
}


def _per_generation_stats(records, instance_num, code_by_id):
    """One dict per finished generation: id, code, validity, per-instance fitness, aggregate."""
    gens = []
    num_gens = len(records) // instance_num
    for g in range(num_gens):
        block = records[g * instance_num:(g + 1) * instance_num]
        algorithm_id = f"algorithm_{g}"
        valid = any(r["success"] for r in block)
        fitnesses = [r["fitness"] for r in block]
        valid_fits = [f for f in fitnesses if f is not None]
        agg = float(np.mean(valid_fits)) if valid_fits else None
        error_type, error = None, None
        for r in block:
            if not r["success"]:
                error_type = r.get("error_type") or "unknown"
                error = r.get("error") or ""
                break
        gens.append({
            "id": algorithm_id,
            "code": code_by_id.get(algorithm_id, ""),
            "valid": valid,
            "fitnesses": fitnesses,
            "agg": agg,
            "error_type": error_type,
            "error": error,
        })
    return gens


def _gen_rank_key(g):
    """Rank generations: prefer more valid instances, then higher mean fitness."""
    valid = [f for f in g["fitnesses"] if f is not None]
    return (len(valid), float(np.mean(valid)) if valid else float("-inf"))


def select_feedback_codes(records, instance_num, code_by_id):
    """Pick previous / last-valid / best generations, collapsing duplicates by id or code."""
    gens = _per_generation_stats(records, instance_num, code_by_id)
    if not gens:
        return []

    previous = gens[-1]
    last_valid = next((g for g in reversed(gens) if g["valid"]), None)
    valid_gens = [g for g in gens if g["valid"]]
    best = max(valid_gens, key=_gen_rank_key) if valid_gens else None

    # Priority order: previous first, then last-valid, then best.
    candidates = [("previous", previous)]
    if last_valid is not None:
        candidates.append(("last_valid", last_valid))
    if best is not None:
        candidates.append(("best", best))

    # Collapse roles that point to the same generation (or to identical code) so
    # we transmit at most three DISTINCT code blocks.
    blocks = []
    for role, g in candidates:
        match = next(
            (b for b in blocks if b["gen"]["id"] == g["id"] or b["gen"]["code"] == g["code"]),
            None,
        )
        if match is not None:
            match["roles"].append(role)
        else:
            blocks.append({"roles": [role], "gen": g})
    return blocks


def _fmt_fitnesses(fitnesses):
    """Render a per-instance fitness vector, e.g. '[12345.0, FAIL, 12410.0]  (mean 12377.5)'."""
    cells = ["{:.1f}".format(f) if f is not None else "FAIL" for f in fitnesses]
    valid = [f for f in fitnesses if f is not None]
    mean = "{:.1f}".format(float(np.mean(valid))) if valid else "n/a"
    return "[" + ", ".join(cells) + "]  (mean " + mean + ")"


def build_reference_section(records, instance_num, code_by_id):
    """Assemble the de-duplicated reference-implementation block (code + fitness) for the prompt."""
    blocks = select_feedback_codes(records, instance_num, code_by_id)
    if not blocks:
        return ""

    parts = [
        f"**Reference implementations ({len(blocks)} shown; identical versions collapsed):**\n"
        "Each block is a prior version followed by its fitness on the SAME "
        f"{instance_num} benchmark instances. Study ALL of them together -- the previous attempt, the "
        "last valid one, and the best so far -- and use them as the basis to produce a NEW "
        "implementation that improves on the best fitness shown. Keep what works in the strongest "
        "version and change what is holding it back."
    ]
    for idx, b in enumerate(blocks, 1):
        g = b["gen"]
        roles = " = ".join(ROLE_LABELS[r] for r in b["roles"])
        if g["valid"]:
            perf = "Fitness per instance: " + _fmt_fitnesses(g["fitnesses"])
        else:
            snippet = (": " + g["error"][:160]) if g["error"] else ""
            perf = f"Result: FAILED ({g['error_type'] or 'unknown'}{snippet})"
        parts.append(
            f"--- Reference {idx} | {roles} | [{g['id']}] ---\n"
            f"{perf}\n"
            f"```python\n{g['code']}\n```"
        )
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #
# Prompt construction (state-aware: cold-start / repair / intensify / explore)
# Up to three previous solutions (previous / last-valid / best) are injected,
# each with its per-instance fitness, deduplicated (H100 has ample context).
# --------------------------------------------------------------------------- #
def get_prompt(i, base_prompt, records, instance_num, code_by_id):
    if i == 0:
        return base_prompt

    summary = summarize_results(records, instance_num)
    feedback_hints = generate_feedback_hints(records, instance_num)
    last_gen_valid = any(r["success"] for r in records[-instance_num:])
    have_valid_base = any(r["success"] for r in records)

    mode, mode_directive = decide_mode(
        records, instance_num, last_gen_valid, have_valid_base
    )

    reference_section = build_reference_section(records, instance_num, code_by_id)

    prompt = (
        f"{base_prompt}\n\n"
        f"=== ITERATION {i} FEEDBACK LOOP ===\n"
        f"You are inside an iterative optimization loop; your previous code was executed and analyzed.\n\n"
        f"**Performance of the most recent generation on {instance_num} benchmark instance(s):**\n{summary}\n\n"
        f"**{mode_directive}**\n\n"
        f"**Strategic guidance:**\n{feedback_hints}\n\n"
        f"{reference_section}\n"
        f"Using the reference implementations above as your basis, produce the next implementation "
        f"following the MODE above and aim to BEAT the best fitness shown. Obey the strict output "
        f"contract: EXACTLY ONE python code block defining main(instance) that returns a permutation "
        f"of range(n); evaluation is external, so do not compute or return any score; keep the dynamic "
        f"seed and self-healing return unchanged; no prose."
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
    INSTANCE_NUM = 5
    NUM_GENERATIONS = 100
    TIMEOUT_SECONDS = 900  # hard kill per instance: safety margin above the 600s (10-min) ceiling

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
    # algorithm_id -> code. Lets the prompt cite the previous / last-valid / best
    # implementations without re-reading them from disk. `records` stays the
    # single source of truth for fitness/validity; until the first valid solution
    # exists, decide_mode falls back to COLD_START.
    code_by_id = {}

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
                code_by_id=code_by_id,
            )

        # --- Generate and extract clean code -------------------------------
        # The skeleton (dynamic seed + self-healing return, NO evaluation
        # primitives) comes from template.py.
        raw = handler.get_response(template_path="./template.py", prompt=prompt)
        code = extract_code(raw)  # clean code from here on (no prose, no fences)
        code_by_id[algorithm_id] = code

        with open(f"algorithms/{algorithm_id}.py", "w") as f:
            f.write(code)

        # --- Evaluate on every instance ------------------------------------
        for j, instance in enumerate(instances):
            tester = LLMhandling.CodeTester(instance=instance, timeout=TIMEOUT_SECONDS)
            result = tester.test(code)

            n = instance.shape[0]
            is_valid = is_valid_solution(result.solution, result.success, n)
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

        # --- Free GPU memory between LLM calls ------------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv("results.csv", index=False)
