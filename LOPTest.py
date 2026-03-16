import numpy as np

import LLMhandling

def fitness_function(solution, LOPInstance):

    total_fitness = 0
    n = len(solution)
    
    for i in range(n):
        for j in range(i + 1, n):
            u = solution[i]
            v = solution[j]
            total_fitness += LOPInstance[u][v]        
    return total_fitness


if __name__ == "__main__":
    model = 'Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8'
    model_args ={
    "temperature": 0.8,
    "max_new_tokens": 2500
    }
    LOPInstance = np.randint(0,100, (100, 100))
    handler = LLMhandling.LLMHandler(mode='local', model_name=model, model_args=model_args)
    tester = LLMhandling.LLMTester(instance = LOPInstance, timeout = 300)
    


