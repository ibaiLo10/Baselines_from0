from litellm import completion
from dotenv import load_dotenv
import multiprocessing

load_dotenv()


class LLMHandler:
    """
    A unified handler for LLM inference supporting:
    - Local vLLM server (OpenAI-compatible API)
    - Remote API providers via litellm

    Args:
        mode (str): 'local' for a local vLLM server, 'api' for external APIs.
        model_name (str): Model name (HF model for vLLM server or API model name).
        model_args (dict | None): Optional configuration dict. Supported keys:
            - temperature (float): Sampling temperature. Default: 0.2.
            - max_new_tokens (int): Maximum tokens to generate. Default: 4096.
        api_base (str | None): Base URL for local vLLM server (default: http://localhost:8000/v1).

    Examples:

        Local usage (vLLM server running):
            >>> handler = LLMHandler(
            ...     mode="local",
            ...     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            ...     api_base="http://localhost:8000/v1",
            ...     model_args={"temperature": 0.2}
            ... )
            >>> response = handler.get_response("Write a binary search function in Python.")

        API usage:
            >>> handler = LLMHandler(
            ...     mode="api",
            ...     model_name="gpt-4o",
            ...     model_args={"temperature": 0.8}
            ... )
            >>> response = handler.get_response("Write a binary search function in Python.")
    """

    def __init__(
        self,
        mode: str = "local",
        model_name: str | None = None,
        model_args: dict | None = None,
        api_base: str | None = None,
    ):
        if model_name is None:
            raise ValueError("model_name is required.")

        self.mode = mode
        self.model_name = model_name
        self.model_args = model_args or {}

        if mode == "local":

            self.api_base = api_base or "http://localhost:8000/v1"

        elif mode == "api":
            self.api_base = api_base

        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'local' or 'api'.")
    
    def apply_template(self, prompt: str, template_path: str) -> str:
        with open(template_path, 'r') as f:
            template_content = f.read()

        prompt = f"""
                You are an expert algorithm designer, your task is to design and create an algorithm to resolve LOP instances.
                The algorithm should be efficient and well-optimized, and it shhould be implemented in Python. The aim of the algorithm
                is to be able to compete if not surpass existing algorithms in terms of solution quality.

                Here is the template you should follow for your response:

                {template_content}

                """
        return prompt


    def get_response(self, prompt_text: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt_text (str): The input prompt to send to the model.

        Returns:
            str: The model's generated response.

        Raises:
            RuntimeError: If the API call fails.
        """
        messages = [{"role": "user", "content": self.apply_template(prompt_text, template_path="template.py")}]

        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=self.model_args.get("temperature", 0.8),
                max_tokens=self.model_args.get("max_new_tokens", 4096),
                api_base=self.api_base
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e


class CodeTester:
    """
    Executes LLM-generated Python code in an isolated subprocess with a timeout.
    Expects the generated code to define a callable named 'main' that accepts a single instance argument.

    Args:
        instance: The input instance passed to the generated 'main' function.
        timeout (int): Maximum seconds to allow for code execution. Default: 5.

    Example:
        >>> tester = CodeTester(instance=my_graph, timeout=5)
        >>> result = tester.test(generated_code_string)
    """

    def __init__(self, instance, timeout: int = 5):
        self.instance = instance
        self.timeout = timeout

    def _run_code(self, queue: multiprocessing.Queue, code: str):
        iso_namespace = {}
        if code.strip().startswith("```"):
            code = code.strip()
            code = code.split("```")
            if len(code) >= 2:
                code = code[1]
            code = code.replace("python", "", 1).strip()

        try:
            byte_code = compile(code, "<string>", "exec")
            exec(byte_code, iso_namespace)

            if "main" not in iso_namespace:
                queue.put("Error: main function wasn't generated.")
                return

            generated_function = iso_namespace["main"]
            result = generated_function(self.instance)
            queue.put(result)

        except Exception as e:
            queue.put(f"Error in generated code: {type(e).__name__}: {e}")

    def test(self, code: str) -> str:
        """
        Run the given code string in an isolated subprocess.

        Args:
            code (str): Python source code string containing a 'main' function.

        Returns:
            str: The result returned by 'main', or an error message.
        """
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_code, args=(queue, code)
        )

        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return f"Error: Execution timed out after {self.timeout} seconds."

        return queue.get() if not queue.empty() else "Error: No result returned."