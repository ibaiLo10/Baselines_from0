import numpy as np
import LLMhandling
import pandas as pd
import os

import LOPbasics

if __name__ == "__main__":
    np.random.seed(42)
    INSTANCE_NUM = 2
    NUM_GENERATIONS = 100
    instances = [np.random.randint(0, 100, (100, 100)) for _ in range(INSTANCE_NUM)]
    model = 'Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8'
    model_args = {
        "temperature": 0.8,
        "max_new_tokens": 2500
    }

    prompt = """
    You are an expert optimization algorithm designer. Your task is to implement an algorithm
    for the Linear Ordering Problem (LOP). Given an n×n matrix, the goal is to find a permutation
    of rows and columns that maximizes the sum of the upper triangle of the reordered matrix.
    The matrix rows and columns are indexed 0 to n-1. The solution must be a permutation: a list
    containing each integer from 0 to n-1 exactly once. The algorithm should be computationally
    efficient and practical for instances of size n=100. Aim for the highest solution quality possible.
    """

    # Load model once — no server needed
    handler = LLMhandling.LLMHandler(mode='hf', model_name=model, model_args=model_args)

    records = []
    os.makedirs("algorithms", exist_ok=True)

    for i in range(NUM_GENERATIONS):
        algorithm_id = f"algorithm_{i}"
        if i == 0:
            code = handler.get_response(template_path="./template.py", prompt=prompt)
        else:
            prompt = f"""
            You are an expert optimization algorithm designer. Your task is to implement an algorithm
            for the Linear Ordering Problem (LOP). Given an n×n matrix, the goal is to find a permutation
            of rows and columns that maximizes the sum of the upper triangle of the reordered matrix.
            The matrix rows and columns are indexed 0 to n-1. The solution must be a permutation: a list
            containing each integer from 0 to n-1 exactly once. The algorithm should be computationally
            efficient and practical for instances of size n=100. Aim for the highest solution quality possible.
            The last code created by an LLM got these results on 100 different instances:
            {records[(i-1)*INSTANCE_NUM : i*INSTANCE_NUM]}
            algorithm_id = the algorithm id which is the same for all 100.
            instance_id = the id of the instance
            fitness = the fitness of the instance
            success = whether the algorithm has returned a solution or not
            error_type = the error type the algorithm got in case it failed
            this is the code the LLM returned:
            {code}
            """
            code = handler.get_response(template_path="./template.py", prompt=prompt)

        with open(f"algorithms/{algorithm_id}.py", "w") as f:
            f.write(code)

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
                "instance_id": j,
                "fitness": fitness,
                "success": is_valid,
                "error_type": result.error_type if not result.success else (
                    None if is_valid else "invalid_solution"),
            })

    df = pd.DataFrame(records)
    df.to_csv("results.csv", index=False)