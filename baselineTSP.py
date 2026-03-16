from transformers import AutoTokenizer, AutoModelForCausalLM
import litellm
from dotenv import load_dotenv
import inspect
import multiprocessing

load_dotenv()

class LLMHandler:
    def __init__(self, mode: str = "local", model_name: str | None = None, model_args: dict | None = None):
        if model_name is None:
            raise ValueError("model_name is required.")
        
        self.mode = mode
        self.model_name = model_name
        self.model_args = model_args or {}
        
        if mode == 'local':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.model.eval()
        elif mode == 'api':
            self.api_model = model_name
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'local' or 'api'.")

    def get_response(self, prompt_text: str) -> str:
        messages = [{"role": "user", "content": prompt_text}]
        
        if self.mode == 'local':
            temperature = self.model_args.get('temperature', 0.2)
            do_sample = temperature > 0
            
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).to(self.model.device)
            
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.model_args.get('max_new_tokens', 1024),
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            return self.tokenizer.decode(
                generated_ids[0][len(input_ids[0]):],
                skip_special_tokens=True
            )
        
        elif self.mode == 'api':
            try:
                response = litellm.completion(
                    model=self.api_model,
                    messages=messages,
                    temperature=self.model_args.get('temperature', 0.2)
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"API call failed: {e}") from e



class CodeTester:
    def __init__(self, code, instance, timeout=5):
        self.code = code
        self.instance = instance
        self.timeout = timeout

    def _run_code(self, queue):
        iso_namespace = {}
        try:
            byte_code = compile(self.code, '<string>', 'exec')
            exec(byte_code, iso_namespace)
            
            if 'main' not in iso_namespace:
                queue.put("Error: main function wasn't generated.")
                return
            
            generated_function = iso_namespace['main']
            result = generated_function(self.instance)
            queue.put(result)
            
        except Exception as e:
            queue.put(f"Error in generated code: {type(e).__name__}: {e}")

    def test(self):
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=self._run_code, args=(queue,))
        
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return f"Error: Execution timed out after {self.timeout} seconds."

        return queue.get() if not queue.empty() else "Error: No result returned."


        


        