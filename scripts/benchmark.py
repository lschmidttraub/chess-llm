"""Benchmark a chess-llm model on random positions.

Generates random chess positions by playing random moves from the starting
position, prompts the model for its best move, and checks legality.

Supports local checkpoints and HuggingFace repos (trained with train.py).

Usage:
    python scripts/benchmark.py --model data/checkpoints --num-positions 200
    python scripts/benchmark.py --model leosct/chess-llm-deepseek-7b --num-positions 500
"""

import argparse
import random
import time

import chess
from tqdm import tqdm
from unsloth import FastLanguageModel

SYSTEM_PROMPT = (
    "You are a chess engine. Given a position in FEN notation, "
    "predict the best next move in UCI notation."
)

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def generate_random_position(min_moves: int = 10, max_moves: int = 80) -> chess.Board | None:
    """Play random legal moves from the start to get a random mid-game position.

    Returns None if the game ends before min_moves.
    """
    board = chess.Board()
    num_moves = random.randint(min_moves, max_moves)

    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None if board.fullmove_number * 2 < min_moves else board
        board.push(random.choice(legal_moves))

    # Ensure the position isn't terminal
    if board.is_game_over():
        return None

    return board


def build_prompt(fen: str) -> str:
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        user=f"Position: {fen}",
    )


def predict_move(model, tokenizer, fen: str) -> str:
    """Run inference and return the raw model output (stripped)."""
    prompt = build_prompt(fen)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.0,
        do_sample=False,
        use_cache=True,
    )
    # Decode only the generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def is_legal_uci(board: chess.Board, move_str: str) -> bool:
    """Check if a UCI string is a legal move on the board."""
    try:
        move = chess.Move.from_uci(move_str)
        return move in board.legal_moves
    except (ValueError, chess.InvalidMoveError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Benchmark chess-llm model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path (local checkpoint dir or HuggingFace repo ID)",
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=200,
        help="Number of random positions to evaluate (default: 200)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=192,
        help="Max sequence length (default: 192)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization (default: True)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--min-moves",
        type=int,
        default=10,
        help="Minimum random moves before sampling position (default: 10)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=80,
        help="Maximum random moves before sampling position (default: 80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    load_in_4bit = not args.no_4bit
    random.seed(args.seed)

    # Load model
    print(f"Loading model: {args.model}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Generate positions and evaluate
    legal_count = 0
    valid_uci_count = 0
    total = 0
    results = []

    print(f"\nEvaluating on {args.num_positions} random positions...\n")
    pbar = tqdm(total=args.num_positions, desc="Benchmarking")

    while total < args.num_positions:
        board = generate_random_position(args.min_moves, args.max_moves)
        if board is None:
            continue

        fen = board.fen()
        predicted = predict_move(model, tokenizer, fen)
        legal = is_legal_uci(board, predicted)

        # Check if output is at least valid UCI format (4-5 chars, valid squares)
        try:
            chess.Move.from_uci(predicted)
            valid_uci = True
        except (ValueError, chess.InvalidMoveError):
            valid_uci = False

        results.append({
            "fen": fen,
            "predicted": predicted,
            "legal": legal,
            "valid_uci": valid_uci,
        })

        if legal:
            legal_count += 1
        if valid_uci:
            valid_uci_count += 1
        total += 1
        pbar.update(1)
        pbar.set_postfix(legal=f"{legal_count / total:.1%}")

    pbar.close()

    # Print results
    accuracy = legal_count / total
    valid_rate = valid_uci_count / total

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model:              {args.model}")
    print(f"Positions evaluated: {total}")
    print(f"Legal moves:         {legal_count}/{total} ({accuracy:.1%})")
    print(f"Valid UCI format:    {valid_uci_count}/{total} ({valid_rate:.1%})")
    print(f"Invalid outputs:    {total - valid_uci_count}/{total} ({1 - valid_rate:.1%})")
    print("=" * 50)

    # Show some examples
    print("\n--- Sample predictions ---")
    illegal = [r for r in results if not r["legal"]]
    legal_examples = [r for r in results if r["legal"]]

    if legal_examples:
        print("\nCorrect (legal) moves:")
        for r in legal_examples[:3]:
            board = chess.Board(r["fen"])
            print(f"  FEN: {r['fen']}")
            print(f"  Predicted: {r['predicted']}")
            print()

    if illegal:
        print("Incorrect (illegal/invalid) outputs:")
        for r in illegal[:3]:
            print(f"  FEN: {r['fen']}")
            print(f"  Predicted: {r['predicted']!r}")
            print()


if __name__ == "__main__":
    main()
