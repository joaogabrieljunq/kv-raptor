# KV-RAPTOR

This is the official repo from the paper KV-RAPTOR: Scalable Tree-Structured Retrieval with KV Cache Compression for Question-Answering Systems.

## Description

**KV-RAPTOR** is a GPU-accelerated variant of the RAPTOR indexing framework, designed to improve Retrieval-Augmented Generation (RAG) quality. It integrates hierarchical retrieval with LM Cache framework to reuse key-value tokens and reduce latency to generate answers.

---

## Key Features

- 🔍 Tree-based hierarchical clustering for scalable retrieval  
- ⚡ GPU acceleration using RAPIDS cuML and CuPy for fast indexing  
- 💾 Integration with LMCache for efficient KV-cache streaming  

---

## Minimal requirements

- Python 3.8+
- GPU with CUDA +11.8.0 support

## License

KV-RAPTOR is released under the MIT License. See the LICENSE file in the repository for full details.
