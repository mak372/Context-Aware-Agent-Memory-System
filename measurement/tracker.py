import csv
import os
from dataclasses import dataclass, field

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class TurnRecord:
    turn: int
    tiered_tokens: int
    naive_tokens: int


@dataclass
class NaiveTracker:
    """Simulates the naive approach: append every message, count cumulative tokens each turn."""
    history: list[dict] = field(default_factory=list)

    def add_turn(self, user_input: str, response: str) -> int:
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        full_text = " ".join(m["content"] for m in self.history)
        return count_tokens(full_text)


def save_results(records: list[TurnRecord], path: str = "results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["turn", "tiered_tokens", "naive_tokens"])
        for r in records:
            writer.writerow([r.turn, r.tiered_tokens, r.naive_tokens])
    print(f"[measurement] Results saved to {path}")


def print_summary(records: list[TurnRecord]):
    print(f"\n{'Turn':>5}  {'Tiered':>10}  {'Naive':>10}  {'Ratio':>8}")
    print("-" * 40)
    for r in records:
        ratio = r.naive_tokens / r.tiered_tokens if r.tiered_tokens > 0 else 0
        print(f"{r.turn:>5}  {r.tiered_tokens:>10}  {r.naive_tokens:>10}  {ratio:>7.1f}x")
    if records:
        final = records[-1]
        savings = 1 - (final.tiered_tokens / final.naive_tokens) if final.naive_tokens > 0 else 0
        print(f"\nFinal turn token savings: {savings:.1%}")
