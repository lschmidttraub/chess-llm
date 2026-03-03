FROM unsloth/unsloth:2026.2.1-pt2.9.0-cu12.8-moe-optimized-training

USER root

# Clone project from GitHub
RUN git clone https://github.com/lschmidttraub/chess-llm.git /workspace/chess-llm

WORKDIR /workspace/chess-llm

# Install project dependencies (python-chess, pyyaml, etc.)
RUN pip install --no-cache-dir python-chess pyyaml tqdm python-dotenv huggingface-hub wandb
