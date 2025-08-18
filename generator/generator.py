import time, threading
import torch
import lmcache_vllm.vllm as vllm
from lmcache_vllm.vllm import LLM
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM


class Generator:
    def __init__(self, model_id, tokenizer, optimized=False):
        self.model = self.load_model(model_id, tokenizer, optimized)

    def load_model(self, model_id, tokenizer, optimized=False):
        if optimized:
            return LLM(
                model=model_id,
                gpu_memory_utilization=0.8,
                enable_chunked_prefill=False,
                max_model_len=32768,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            ).to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_answer(self, prompt, experiment_name):
        """
        Generate an answer from a prompt using the loaded model.

        Returns:
            answer (str)
            tfft (float): time to first token
            e2e_latency (float): end-to-end latency
            itl (float): inter-token latency
            tps (float): tokens per second
            token_count (int): number of tokens generated
        """
        tokenizer = self.tokenizer
        model = self.model

        if "cache" in experiment_name:
            sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=128)
            generation_start = time.time()
            outputs = model.generate([prompt], sampling_params)
            generation_end = time.time()

            request_output = outputs[0]
            metrics_obj = request_output.metrics
            tfft = metrics_obj.first_token_time - metrics_obj.arrival_time
            answer = request_output.outputs[0].text.strip()
            e2e_latency = generation_end - generation_start
            token_count = len(tokenizer.encode(answer))
            itl = (e2e_latency / token_count) if token_count > 1 else 0.0
            tps = (token_count / e2e_latency) if e2e_latency > 0 else 0.0

        else:
            start_time_sync = time.time()
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            generation_thread = threading.Thread(
                target=model.generate,
                kwargs={
                    "inputs": inputs["input_ids"],
                    "max_new_tokens": 128,
                    "streamer": streamer,
                },
            )
            generation_thread.start()
            token_times = []
            generated_text = ""
            for token in streamer:
                now = time.time()
                token_times.append(now)
                if len(token_times) == 1:
                    tfft = now - start_time_sync
                generated_text += token
            generation_thread.join()
            answer = generated_text.strip()
            end_time_sync = time.time()
            e2e_latency = end_time_sync - start_time_sync
            if len(token_times) > 1:
                diffs = [t2 - t1 for t1, t2 in zip(token_times[:-1], token_times[1:])]
                itl = sum(diffs) / len(diffs)
                generation_time = token_times[-1] - token_times[0]
                tps = len(token_times) / generation_time if generation_time > 0 else 0.0
            else:
                itl = 0.0
                tps = 0.0
            token_count = len(tokenizer.encode(answer))

        return answer, tfft, e2e_latency, itl, tps, token_count
