# Mini-LLaMA From Scratch

Train a modern LLaMA-style language model from scratch in PyTorch. This project implements the same architecture used in LLaMA 3 and Mistral — built as an interactive Jupyter notebook for learning and portfolio use.

## Architecture

This is not a toy GPT. The model uses the same modern components found in production LLMs:

| Component | What We Use | What It Replaces |
|-----------|------------|-----------------|
| Normalization | **RMSNorm** (pre-norm) | LayerNorm |
| Position Encoding | **RoPE** (Rotary Position Embeddings) | Learned / Sinusoidal |
| Activation | **SwiGLU** | ReLU / GELU |
| Attention | **Grouped Query Attention (GQA)** | Multi-Head Attention |
| Bias | **None** | Bias in all linear layers |

## Model Configs

Three sizes tuned for consumer GPUs:

| Config | Parameters | Layers | Dim | Heads | KV Heads | VRAM |
|--------|-----------|--------|-----|-------|----------|------|
| Small | ~15M | 6 | 384 | 6 | 2 | ~2 GB |
| Medium | ~45M | 8 | 512 | 8 | 2 | ~4 GB |
| Large | ~110M | 12 | 768 | 12 | 4 | ~8 GB |

## What You'll Learn

- Building a transformer from scratch — every layer explained
- RoPE: how rotary embeddings encode relative position
- GQA: how sharing KV heads reduces memory with minimal quality loss
- SwiGLU: why gated activations outperform ReLU/GELU
- Training with mixed precision (AMP), cosine LR schedule, gradient clipping
- AdamW with decoupled weight decay and parameter group separation
- Attention pattern visualization
- Top-k sampling for text generation

## Quick Start

```bash
git clone https://github.com/thejaredchapman/pytorch-llm-lab.git
cd pytorch-llm-lab
pip install -r requirements.txt
jupyter notebook mini_llama_from_scratch.ipynb
```

## Requirements

- Python 3.9+
- PyTorch 2.1+ (for `scaled_dot_product_attention`)
- NVIDIA GPU recommended (tested on 3080 Ti 12GB)
- ~2-3 GB disk space

## Notebook Sections

1. **Setup** — GPU detection, imports
2. **Data** — Auto-downloads Shakespeare (~1MB)
3. **Tokenizer** — Character-level encoder/decoder
4. **Data Loader** — Random batch sampling with train/val split
5. **RMSNorm** — Implementation with explanation
6. **RoPE** — Rotary embeddings with frequency visualization
7. **Grouped Query Attention** — Full GQA with RoPE integration
8. **SwiGLU FFN** — Gated feed-forward network
9. **Transformer Block** — Pre-norm residual architecture
10. **Full Model** — MiniLLaMA with weight tying and generation
11. **Training Loop** — AdamW, cosine LR, AMP, gradient clipping
12. **Visualization** — Loss curves, perplexity, LR schedule
13. **Generation Playground** — Interactive text generation with temperature control
14. **Attention Visualization** — See what the heads learned
15. **Interview Prep** — Q&A covering every architectural decision

## Exercises Included

- Scale from Small → Large and compare loss curves
- Hyperparameter search (LR, batch size, sequence length)
- Swap GQA for standard MHA and measure the difference
- Add dropout and analyze the train/val gap
- Train on your own dataset

## License

MIT
