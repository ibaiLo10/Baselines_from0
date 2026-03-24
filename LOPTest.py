import numpy as np
import subprocess
import time
import requests
import LLMhandling


def start_vllm_server(model: str, port: int = 8000):
    process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            requests.get(f"http://localhost:{port}/health")
            print("Server ready.")
            return process
        except requests.ConnectionError:
            time.sleep(5)
    raise RuntimeError("vLLM server did not start in time.")


def fitness_function(solution, lop_instance):
    total_fitness = 0
    n = len(solution)

    for i in range(n):
        for j in range(i + 1, n):
            u = solution[i]
            v = solution[j]
            total_fitness += lop_instance[u][v]
    return total_fitness


if __name__ == "__main__":
    model = 'Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8'
    model_args = {
        "temperature": 0.8,
        "max_new_tokens": 2500
    }
    port = 14001
    base = f"http://localhost:{port}"
    LOPInstance = np.random.randint(0, 100, (100, 100))
    server = start_vllm_server(model, )
    try:
        handler = LLMhandling.LLMHandler(mode='local', model_name=model, model_args=model_args, api_base=base)
        tester = LLMhandling.CodeTester(instance=LOPInstance, timeout=300)
        code = None
        fitness_list = []
        for i in range(100):
            code = handler.get_response(template_path="./template.py", sampling_mode='iid')
            test = tester.test(code)
    finally:
        server.terminate()