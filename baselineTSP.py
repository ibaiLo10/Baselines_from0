from litellm import completion
from dotenv import load_dotenv
import multiprocessing

load_dotenv()


class LLMHandler:
    """
    A unified handler for LLM inference supporting local (vLLM) and API (litellm) modes.

    Args:
        mode (str): 'local' for vLLM inference, 'api' for litellm API calls.
        model_name (str): HuggingFace model ID for local mode (e.g. 'Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8')
                          or litellm model string for API mode (e.g. 'gpt-4o', 'anthropic/claude-3-5-sonnet').
        model_args (dict | None): Optional configuration dict. Supported keys:
            - temperature (float): Sampling temperature. Default: 0.2.
            - max_new_tokens (int): Maximum tokens to generate. Default: 4096. (local only)

    Raises:
        ValueError: If model_name is not provided or mode is invalid.
        ImportError: If vllm is not installed and mode is 'local'.

    Examples:
        Local usage (requires vllm and a GPU):
            >>> handler = LLMHandler(
            ...     mode="local",
            ...     model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            ...     model_args={"temperature": 0.2, "max_new_tokens": 4096}
            ... )
            >>> response = handler.get_response("Write a binary search function in Python.")

        API usage (requires appropriate API key in .env):
            >>> handler = LLMHandler(
            ...     mode="api",
            ...     model_name="gpt-4o",
            ...     model_args={"temperature": 0.2}
            ... )
            >>> response = handler.get_response("Write a binary search function in Python.")
    """

    def __init__(self, mode: str = "local", model_name: str | None = None, model_args: dict | None = None):
        if model_name is None:
            raise ValueError("model_name is required.")

        self.mode = mode
        self.model_name = model_name
        self.model_args = model_args or {}

        if mode == 'local':
            try:
                from vllm import LLM, SamplingParams
                self._SamplingParams = SamplingParams
            except ImportError:
                raise ImportError(
                    "vllm is required for local mode. "
                    "Install it with: pip install vllm"
                )
            self.model = LLM(model=model_name)

        elif mode == 'api':
            self.api_model = model_name

        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'local' or 'api'.")

    def get_response(self, prompt_text: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt_text (str): The input prompt to send to the model.

        Returns:
            str: The model's generated response.

        Raises:
            RuntimeError: If the API call fails in 'api' mode.
        """
        messages = [{"role": "user", "content": prompt_text}]

        if self.mode == 'local':
            sampling_params = self._SamplingParams(
                temperature=self.model_args.get('temperature', 0.2),
                max_tokens=self.model_args.get('max_new_tokens', 4096),
            )
            outputs = self.model.chat(messages, sampling_params=sampling_params)
            return outputs[0].outputs[0].text

        elif self.mode == 'api':
            try:
                response = completion(
                    model=self.api_model,
                    messages=messages,
                    temperature=self.model_args.get('temperature', 0.2)
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"API call failed: {e}") from e


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
        try:
            byte_code = compile(code, '<string>', 'exec')
            exec(byte_code, iso_namespace)
            if 'main' not in iso_namespace:
                queue.put("Error: main function wasn't generated.")
                return
            generated_function = iso_namespace['main']
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
        process = multiprocessing.Process(target=self._run_code, args=(queue, code))
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return f"Error: Execution timed out after {self.timeout} seconds."

        return queue.get() if not queue.empty() else "Error: No result returned."