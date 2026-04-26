from litellm import completion
from dotenv import load_dotenv
import multiprocessing
import re
import time
import requests
import subprocess
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


def _load_hf_model(model_name: str, model_args: dict):
    """
    Load a HuggingFace model and tokenizer directly for local inference.
    Tries inference_mode first, falls back to no_grad.
    Returns (model, tokenizer, context_manager).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required for 'hf' mode. "
            "Install with: pip install transformers torch"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[HFHandler] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[HFHandler] Loading model on {device} with dtype {dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    model.eval()

    # Prefer inference_mode (no autograd overhead), fall back to no_grad
    try:
        torch.inference_mode()
        ctx = torch.inference_mode
        print("[HFHandler] Using torch.inference_mode()")
    except AttributeError:
        ctx = torch.no_grad
        print("[HFHandler] torch.inference_mode not available, using torch.no_grad()")

    return model, tokenizer, ctx, device


class LLMHandler:
    """
    A unified handler for LLM inference supporting:
    - 'hf'    : Direct HuggingFace transformers inference (no server needed).
                Uses torch.inference_mode() or torch.no_grad() automatically.
    - 'local' : Local vLLM server (OpenAI-compatible API).
    - 'api'   : Remote API providers via litellm.

    Args:
        mode (str): 'hf' | 'local' | 'api'.
        model_name (str): HF model repo id, vLLM model name, or API model name.
        model_args (dict | None): Optional config. Supported keys:
            - temperature (float): Sampling temperature. Default: 0.8.
            - max_new_tokens (int): Maximum tokens to generate. Default: 4096.
        api_base (str | None): Base URL for local vLLM server (only used in 'local' mode).

    Examples:

        HF direct inference (no server):
            >>> handler = LLMHandler(
            ...     mode="hf",
            ...     model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
            ...     model_args={"temperature": 0.8, "max_new_tokens": 2500}
            ... )
            >>> response = handler.get_response("./template.py", "Write a sort algorithm.")

        Local vLLM server:
            >>> handler = LLMHandler(
            ...     mode="local",
            ...     model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            ...     api_base="http://localhost:8000/v1",
            ...     model_args={"temperature": 0.8}
            ... )

        Remote API:
            >>> handler = LLMHandler(
            ...     mode="api",
            ...     model_name="gpt-4o",
            ...     model_args={"temperature": 0.8}
            ... )
    """

    def __init__(
            self,
            mode: str = "hf",
            model_name: str | None = None,
            model_args: dict | None = None,
            api_base: str | None = None,
    ):
        if model_name is None:
            raise ValueError("model_name is required.")

        self.mode = mode
        self.model_name = model_name
        self.model_args = model_args or {}

        # HF mode: load model once at construction time
        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_ctx = None
        self._hf_device = None

        if mode == "hf":
            self._hf_model, self._hf_tokenizer, self._hf_ctx, self._hf_device = \
                _load_hf_model(model_name, self.model_args)
        elif mode == "local":
            self.api_base = api_base or "http://localhost:8000/v1"
        elif mode == "api":
            self.api_base = api_base
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'hf', 'local', or 'api'.")

    def apply_template(self, template_path: str, instruction: str) -> str:
        with open(template_path, 'r') as f:
            template_content = f.read()
        return f"{instruction}\n{template_content}"

    def _generate_hf(self, prompt: str) -> str:
        """Run inference directly through the loaded HF model."""
        tokenizer = self._hf_tokenizer
        model = self._hf_model
        ctx = self._hf_ctx
        device = self._hf_device

        # Build chat messages if tokenizer supports apply_chat_template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        max_new_tokens = self.model_args.get("max_new_tokens", 4096)
        temperature = self.model_args.get("temperature", 0.8)
        do_sample = temperature > 0.0

        with ctx():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    def get_response(self, template_path: str, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            template_path (str): Path to the template/skeleton the LLM should follow.
            prompt (str): Instructions for the task.

        Returns:
            str: The model's generated response.

        Raises:
            RuntimeError: If inference fails.
        """
        full_prompt = self.apply_template(template_path=template_path, instruction=prompt)

        try:
            if self.mode == "hf":
                return self._generate_hf(full_prompt)

            # 'local' or 'api' — use litellm
            messages = [{"role": "user", "content": full_prompt}]
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=self.model_args.get("temperature", 0.8),
                max_tokens=self.model_args.get("max_new_tokens", 4096),
                api_base=getattr(self, "api_base", None),
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


def start_vllm_server(model: str, port: int = 8000, timeout_minutes: int = 15):
    """
    Kept for backwards compatibility. Prefer mode='hf' in LLMHandler instead.
    """
    log_file = open("vllm_server.log", "w")
    process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
        ],
        stdout=log_file,
        stderr=log_file,
    )

    attempts = timeout_minutes * 2  # check every 30s
    for i in range(attempts):
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(
                f"vLLM server process exited with code {process.returncode}. "
                f"Check vllm_server.log for details."
            )
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if r.status_code == 200:
                print("Server ready.")
                return process
        except requests.ConnectionError:
            pass

        print(f"Waiting for vLLM server... ({(i+1)*30}s / {timeout_minutes*60}s)")
        time.sleep(30)

    log_file.flush()
    raise RuntimeError(
        f"vLLM server did not start within {timeout_minutes} minutes. "
        f"Check vllm_server.log for details."
    )