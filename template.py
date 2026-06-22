"""
Fill in ONLY the TODO section below. Do not change the signature, the import
markers, the `solution` variable, or the final `return`.
Add every import you need between the "#Imports start" / "#Imports end" markers,
and implement every helper function/class INSIDE main().
"""
def main(instance):
    #Imports start
    import time
    import random
    import numpy as np
    #Imports end
    """
    - instance: a NumPy array of shape (n, n).
    - Return a Python list of n DISTINCT integers: a permutation of range(n).
      NOT a NumPy array. No duplicates, no missing indices, length exactly n.
    - Implement all helpers inside main(). Do not print anything.
    - Only fill the TODO; keep the rest unchanged.
    - The time limit must be 10 minutes, which means a valid solution must be
      returned within that time.
    """

    # Dynamic seed: the nanosecond clock makes every run explore differently, so
    # the SAME generated algorithm is not deterministic across executions. Masked
    # to 32 bits to stay inside the range accepted by np.random.seed / random.seed,
    # and applied to BOTH RNGs because generated code may use either.
    _seed = time.time_ns() & 0xFFFFFFFF
    np.random.seed(_seed)
    random.seed(_seed)

    # Self-healing default: a valid permutation of range(n) is in place from the
    # very start, so an early time-out (or any branch that never assigns) still
    # returns something legal. The TODO must overwrite this with its best result.
    solution = list(range(instance.shape[0]))
    TIME_LIMIT = 600  # the limit is set in seconds.
    #TODO: implement the algorithm here. Assign the result to `solution`.
    #END TODO
    return solution
