import os
import time
import pandas as pd


class Evaluator:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.responses = []

        self.all_tfft = []
        self.all_e2e_latency = []
        self.all_itl = []
        self.all_tps = []
        self.all_retriever_time = []
        self.all_input_context_size = []
        self.all_output_tokens_size = []

        self.start_time = time.time()
        self.date_str = time.strftime("%d-%m-%Y")
        self.folder_name = f"{experiment_name}_{self.date_str}"
        os.makedirs(self.folder_name, exist_ok=True)
        self.csv_filename = os.path.join(self.folder_name, f"queries_results_{experiment_name}_{self.date_str}.csv")
        self.metrics_filename = os.path.join(self.folder_name, f"overall_metrics_{experiment_name}_{self.date_str}.txt")

    def record(self, query, context, answer, expected_output, tfft, e2e_latency, itl, tps, retriever_time, input_context_size, output_tokens_size):
        self.all_tfft.append(tfft)
        self.all_e2e_latency.append(e2e_latency)
        self.all_itl.append(itl)
        self.all_tps.append(tps)
        self.all_retriever_time.append(retriever_time)
        self.all_input_context_size.append(input_context_size)
        self.all_output_tokens_size.append(output_tokens_size)

        self.responses.append({
            "query": query,
            "context": context,
            "answer": answer,
            "expected_output": expected_output,
            "tfft": tfft,
            "e2e_latency": e2e_latency,
            "itl": itl,
            "tps": tps,
            "retriever_time": retriever_time,
            "input_context_size": input_context_size,
            "output_tokens_size": output_tokens_size
        })

    def finalize(self):
        total_time = time.time() - self.start_time
        n = len(self.responses)

        metrics = {
            "rps": n / total_time if total_time > 0 else 0.0,
            "avg_tfft": sum(self.all_tfft) / n if n else 0.0,
            "avg_e2e": sum(self.all_e2e_latency) / n if n else 0.0,
            "avg_itl": sum(self.all_itl) / n if n else 0.0,
            "avg_tps": sum(self.all_tps) / n if n else 0.0,
            "avg_retriever": sum(self.all_retriever_time) / n if n else 0.0,
            "avg_input_ctx": sum(self.all_input_context_size) / n if n else 0.0,
            "avg_output_tokens": sum(self.all_output_tokens_size) / n if n else 0.0
        }

        metrics_str = (
            f"Overall RPS: {metrics['rps']:.2f} req/sec, "
            f"Average TTFT: {metrics['avg_tfft']:.2f} sec, "
            f"Average E2E Latency: {metrics['avg_e2e']:.2f} sec, "
            f"Average ITL: {metrics['avg_itl']:.2f} sec, "
            f"Average TPS: {metrics['avg_tps']:.2f} tokens/sec, "
            f"Average Retriever Time: {metrics['avg_retriever']:.2f} sec, "
            f"Average Input Context Size: {metrics['avg_input_ctx']:.0f} tokens, "
            f"Average Output Tokens Size: {metrics['avg_output_tokens']:.0f} tokens"
        )

        print("\nOverall Metrics:")
        print(metrics_str)

        # Save to text
        with open(self.metrics_filename, "w") as f:
            f.write(metrics_str)
        print(f"Saved overall metrics to {self.metrics_filename}")

        # Save to CSV
        df = pd.DataFrame(self.responses)
        df.to_csv(self.csv_filename, index=False)
        print(f"Saved detailed results to {self.csv_filename}")

        return self.responses