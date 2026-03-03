FROM unsloth/unsloth:2026.2.1-pt2.9.0-cu12.8-moe-optimized-training

USER root

# Copy project files
COPY configs/ /workspace/chess-llm/configs/
COPY scripts/ /workspace/chess-llm/scripts/
COPY pyproject.toml /workspace/chess-llm/

# Install project dependencies (python-chess, pyyaml, etc.)
RUN pip install --no-cache-dir python-chess pyyaml tqdm python-dotenv huggingface-hub wandb

WORKDIR /workspace/chess-llm
