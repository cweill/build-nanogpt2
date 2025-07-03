# Build NanoGPT2

This project is an implementation of GPT-2 (124M parameters) based on the YouTube video **"Let's reproduce GPT-2 (124M)"** by Andrej Karpathy.

## About

This is a PyTorch implementation of a smaller version of GPT-2, designed for educational purposes and to understand the architecture and training process of transformer-based language models.

## Video Reference

The implementation follows along with Andrej Karpathy's tutorial:
- **Video**: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=11&ab_channel=AndrejKarpathy)
- **Channel**: Andrej Karpathy
- **Series**: Neural Networks: Zero to Hero

## Project Structure

- `train_gpt2.py` - Main training script with GPT-2 model implementation
- `play.ipynb` - Jupyter notebook for experimentation and testing
- `pyproject.toml` - Python project configuration and dependencies

## Model Configuration

The current implementation uses a smaller configuration:
- Block size: 256
- Vocabulary size: 65
- Number of layers: 6
- Number of attention heads: 6
- Embedding dimension: 384

## Getting Started

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the training script:
   ```bash
   uv run train_gpt2.py
   ```

## License

This project is for educational purposes. The original GPT-2 model and architecture are from OpenAI.
