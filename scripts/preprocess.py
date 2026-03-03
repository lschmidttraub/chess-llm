"""Preprocess chess data into a HuggingFace Dataset for LLM fine-tuning.

Supports two data sources:
- local: Parse PGN files from data/ directory
- remote: Download GambitFlow/Elite-Data SQLite database from HuggingFace

Output format: Each example is a complete chat-template-formatted string with
a FEN position as input and the best next move (UCI) as output.
"""

import argparse
import io
import sqlite3
from pathlib import Path

import chess
import chess.pgn
from datasets import Dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a chess engine. Given a position in FEN notation, "
    "predict the best next move in UCI notation."
)

# ChatML template matching Qwen/DeepSeek models
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)


def format_example(fen: str, move_uci: str) -> str:
    return CHAT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        user=f"Position: {fen}",
        assistant=move_uci,
    )


def extract_from_pgn(pgn_path: Path, max_samples: int | None, skip_first_n_moves: int = 10) -> list[dict]:
    """Extract (FEN, next_move) pairs from a PGN file."""
    examples = []
    with open(pgn_path) as f:
        pbar = tqdm(desc=f"Parsing {pgn_path.name}", unit=" positions")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            for move_num, move in enumerate(game.mainline_moves()):
                if move_num >= skip_first_n_moves:
                    fen = board.fen()
                    move_uci = move.uci()
                    examples.append({
                        "text": format_example(fen, move_uci),
                        "fen": fen,
                        "move": move_uci,
                    })
                    pbar.update(1)

                    if max_samples and len(examples) >= max_samples:
                        pbar.close()
                        return examples

                board.push(move)

        pbar.close()
    return examples


def extract_from_elite_data(max_samples: int | None) -> list[dict]:
    """Download and extract positions from GambitFlow/Elite-Data."""
    print("Downloading GambitFlow/Elite-Data from HuggingFace...")
    db_path = hf_hub_download(
        repo_id="GambitFlow/Elite-Data",
        filename="match_positions_v2.db",
        repo_type="dataset",
    )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    limit_clause = f" LIMIT {max_samples}" if max_samples else ""
    cursor.execute(
        f"SELECT fen, move_played FROM positions WHERE move_played IS NOT NULL{limit_clause}"
    )

    examples = []
    for fen, move_played in tqdm(cursor.fetchall(), desc="Processing Elite-Data"):
        examples.append({
            "text": format_example(fen, move_played),
            "fen": fen,
            "move": move_played,
        })

    conn.close()
    return examples


def main():
    parser = argparse.ArgumentParser(description="Preprocess chess data for fine-tuning")
    parser.add_argument(
        "--source",
        choices=["local", "remote", "both"],
        default="local",
        help="Data source: local PGN, remote Elite-Data, or both",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to extract (per source)",
    )
    parser.add_argument(
        "--pgn-path",
        type=Path,
        default=Path("data/lichess_elite_2025-11.pgn"),
        help="Path to local PGN file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--skip-opening-moves",
        type=int,
        default=10,
        help="Number of opening moves to skip per game",
    )
    args = parser.parse_args()

    examples = []

    if args.source in ("local", "both"):
        if not args.pgn_path.exists():
            print(f"PGN file not found: {args.pgn_path}")
            if args.source == "local":
                return
        else:
            examples.extend(extract_from_pgn(args.pgn_path, args.max_samples, args.skip_opening_moves))
            print(f"Extracted {len(examples)} positions from local PGN")

    if args.source in ("remote", "both"):
        remote_examples = extract_from_elite_data(args.max_samples)
        examples.extend(remote_examples)
        print(f"Extracted {len(remote_examples)} positions from Elite-Data")

    if not examples:
        print("No examples extracted. Exiting.")
        return

    print(f"Total examples: {len(examples)}")

    dataset = Dataset.from_list(examples)
    dataset = dataset.shuffle(seed=42)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    print(f"Dataset saved to {args.output_dir}")

    # Print a sample
    print("\n--- Sample ---")
    print(dataset[0]["text"])


if __name__ == "__main__":
    main()
