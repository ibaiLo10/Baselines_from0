from litellm import completion
from dotenv import load_dotenv
import multiprocessing
import re
import traceback
from dataclasses import dataclass
from typing import Any

load_dotenv()


@dataclass
class TestResult:
    """
    Structured result returned by CodeTester.test().

    Attributes:
        success    : True only if execution completed without errors.
        solution   : The raw value returned by main(), or None on failure.
        error      : Human-readable error message, or None on success.
        error_type : One of 'empty' | 'timeout' | 'compile' | 'runtime' | 'no_result' | None.
        traceback  : Full traceback string when available, else None.
    """
    success: bool
    solution: Any = None
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None


def _strip_markdown_fences(code: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers if present."""
    code = code.strip()
    m = re.compile(r'^```[a-zA-Z]*\n?(.*?)```$', re.DOTALL).match(code)
    return m.group(1).strip() if m else code


def _subprocess_entry(queue: multiprocessing.Queue, code: str, instance):
    """
    Entry point executed inside the isolated subprocess.
    Always puts a result dict onto the queue — never raises.
    """
    result = {
        "success": False, "solution": None,
        "error": None, "error_type": None, "traceback": None
    }

    clean_code = _strip_markdown_fences(code)

    # Compile
    try:
        byte_code = compile(clean_code, "<llm_generated>", "exec")
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        result["error_type"] = "compile"
        result["traceback"] = traceback.format_exc()
        queue.put(result)
        return

    # Execute module-level (imports, definitions)
    iso_namespace = {}
    try:
        exec(byte_code, iso_namespace)
    except Exception as e:
        result["error"] = f"Error during module-level execution: {type(e).__name__}: {e}"
        result["error_type"] = "runtime"
        result["traceback"] = traceback.format_exc()
        queue.put(result)
        return

    # Check 'main' exists and is callable
    if "main" not in iso_namespace or not callable(iso_namespace["main"]):
        result["error"] = "main' function not found or not callable in generated code."
        result["error_type"] = "runtime"
        queue.put(result)
        return

    # Call main(instance)
    try:
        solution = iso_namespace["main"](instance)
        result.update(success=True, solution=solution)
    except Exception as e:
        result["error"] = f"Error inside main(): {type(e).__name__}: {e}"
        result["error_type"] = "runtime"
        result["traceback"] = traceback.format_exc()

    queue.put(result)


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
            ...     model_args={"temperature": 0.8}
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

    def apply_template(self, template_path: str, instruction: str) -> str:
        with open(template_path, 'r') as f:
            template_content = f.read()
        prompt = f"{instruction}\n{template_content}"

        return prompt

    def get_response(self, template_path: str , prompt: str ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            template_path (str): The input path of the template/skeleton the LLM should follow.
            prompt (str):  Instructions given to the LLM to indicate what the task will be.

        Returns:
            str: The model's generated response.

        Raises:
            RuntimeError: If the API call fails.
        """
        messages = [{"role": "user",
                     "content": self.apply_template(template_path=template_path, instruction = prompt)}]

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
    Expects the generated code to define a callable named 'main' that accepts
    a single instance argument.

    Args:
        instance : The input instance passed to the generated 'main' function.
        timeout  : Max seconds allowed for execution. Default: 30.

    Example:
        >>> tester = CodeTester(instance=my_matrix, timeout=30)
        >>> result = tester.test(generated_code_string)
        >>> if result.success:
        ...     print("Solution:", result.solution)
        ... else:
        ...     print(result.error_type, result.error)
    """

    def __init__(self, instance, timeout: int = 30):
        self.instance = instance
        self.timeout = timeout

    def test(self, code: str) -> TestResult:
        """
        Run *code* in an isolated subprocess and return a TestResult.

        Args:
            code : Python source string (raw or markdown-fenced).

        Returns:
            TestResult — always returns structurally; check .success for outcome.
        """
        if not code or not code.strip():
            return TestResult(
                success=False,
                error="Empty code string received.",
                error_type="empty",
            )

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(
            target=_subprocess_entry,
            args=(queue, code, self.instance),
        )

        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return TestResult(
                success=False,
                error=f"Execution timed out after {self.timeout} seconds.",
                error_type="timeout",
            )

        if queue.empty():
            return TestResult(
                success=False,
                error="Subprocess exited without returning a result (possible crash or OOM).",
                error_type="no_result",
            )

        return TestResult(**queue.get_nowait())