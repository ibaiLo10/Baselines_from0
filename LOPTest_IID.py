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
    port = 14001
    base = f"http://localhost:{port}/v1"

    prompt = """
    You are an expert optimization algorithm designer. Your task is to implement an algorithm
    for the Linear Ordering Problem (LOP). Given an n×n matrix, the goal is to find a permutation
    of rows and columns that maximizes the sum of the upper triangle of the reordered matrix.
    The matrix rows and columns are indexed 0 to n-1. The solution must be a permutation: a list
    containing each integer from 0 to n-1 exactly once. The algorithm should be computationally
    efficient and practical for instances of size n=100. Aim for the highest solution quality possible.
    """

    server = LLMhandling.start_vllm_server(model, port)
    records = []

    try:
        handler = LLMhandling.LLMHandler(mode='local', model_name=model, model_args=model_args, api_base=base)
        os.makedirs("algorithms", exist_ok=True)
        for i in range(NUM_GENERATIONS):
            algorithm_id = f"algorithm_{i}"
            code = handler.get_response(template_path="./template.py", prompt=prompt)
            with open(f"algorithms/{algorithm_id}.py", "w") as f:
                f.write(code)

            for j, instance in enumerate(instances):
                tester = LLMhandling.CodeTester(instance=instance, timeout=300)
                result = tester.test(code)

                if result.success:
                    fitness = LOPbasics.fitness_function(result.solution, instance)
                else:
                    fitness = None

                records.append({
                    "algorithm_id": algorithm_id,
                    "instance_id": j,
                    "fitness": fitness,
                    "success": result.success,
                    "error_type": result.error_type,
                })

    finally:
        server.terminate()
        df = pd.DataFrame(records)
        df.to_csv("results.csv", index=False)