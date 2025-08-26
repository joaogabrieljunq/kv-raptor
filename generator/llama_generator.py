import os, time, threading, datetime
import torch
import lmcache
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


class Generator:
    def __init__(self, optimized=False, use_disk=False):
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = self.load_model(self.model_id, optimized, use_disk)
        self.optimized = optimized

    def setup_lmcache_env(self, use_disk: bool = False):
        os.environ["LMCACHE_CHUNK_SIZE"] = "256"
        if use_disk:
            os.environ["LMCACHE_LOCAL_CPU"] = "False"
            os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
            os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
            os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
        else:
            os.environ["LMCACHE_LOCAL_CPU"] = "True"
            os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

    def load_model(self, model_id, optimized=False, use_disk=False):
        if optimized:
            self.setup_lmcache_env(use_disk)
            kv_config = KVTransferConfig(
                kv_connector="LMCacheConnectorV1",
                kv_role="kv_both",
            )
            return LLM(
                model=model_id,
                gpu_memory_utilization=0.8,
                max_model_len=32768,
                kv_transfer_config=kv_config,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            ).to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_answer(self, query, context, experiment_name):
        model = self.model
        today_date_verbose = datetime.date.today().strftime("%d %B %Y")
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"Cutting Knowledge Date: December 2023\n"
            f"Today Date: {today_date_verbose}\n\n"
            "You are a helpful assistant. Please provide a concise and accurate answer in one short sentence or noun based on the context.\n"
            "Use the provided context only, do not invent.\n"
            "Avoid repeating yourself\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Question: {query}\n"
            f"Context:\n{context}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        # Tokenize to get input context size.
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_context_size = inputs.input_ids.shape[1]

        if self.optimized and "cache" in experiment_name:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
            generation_start = time.time()
            outputs = model.generate([prompt], sampling_params)
            generation_end = time.time()

            request_output = outputs[0]
            answer = request_output.outputs[0].text.strip()
            e2e_latency = generation_end - generation_start
            token_count = len(self.tokenizer.encode(answer))

            # TTFT manual: estimado como e2e_latency / token_count
            tfft = (e2e_latency / token_count) if token_count > 0 else 0.0
            itl = tfft  # first token estimate = avg latency per token
            tps = (token_count / e2e_latency) if e2e_latency > 0 else 0.0

        else:
            start_time_sync = time.time()
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(
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
            token_count = len(self.tokenizer.encode(answer))

        return answer, tfft, e2e_latency, itl, tps, input_context_size, token_count