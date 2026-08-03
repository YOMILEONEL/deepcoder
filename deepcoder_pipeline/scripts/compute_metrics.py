import argparse
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="results_T2_gas10000.csv")
parser.add_argument("--outfile", type=str, default="results_T2_gas10000_with_metrics.csv")
args, _ = parser.parse_known_args()

INPUT_FILE = args.infile
OUTPUT_FILE = args.outfile


def tokenize_program(program):
    """
    Converts a DeepCoder program string into tokens.
    Example:
    LIST|LIST|COUNT,>0,0|ZIPWITH,+,1,0
    ->
    ["LIST", "LIST", "COUNT,>0,0", "ZIPWITH,+,1,0"]
    """
    if pd.isna(program):
        return []

    program = str(program).strip()

    if program == "" or program.lower() in {"none", "nan"}:
        return []

    return [token.strip() for token in program.split("|") if token.strip()]


def accuracy(reference_tokens, solution_tokens):
    """
    Binary Accuracy:
    1 if the generated program is exactly equal to the reference program,
    otherwise 0.
    """
    return 1.0 if reference_tokens == solution_tokens else 0.0


def program_operation_score(reference_tokens, solution_tokens):
    """
    Measures whether the same operations/tokens occur,
    independent of their position.

    It uses multiset overlap, so repeated operations are handled correctly.
    """
    if not reference_tokens:
        return 0.0

    ref_counter = Counter(reference_tokens)
    sol_counter = Counter(solution_tokens)

    overlap = 0
    for token in ref_counter:
        overlap += min(ref_counter[token], sol_counter.get(token, 0))

    return overlap / len(reference_tokens)


def program_position_score(reference_tokens, solution_tokens):
    """
    Measures how many tokens are equal at the same position.
    """
    if not reference_tokens:
        return 0.0

    max_len = max(len(reference_tokens), len(solution_tokens))
    if max_len == 0:
        return 0.0

    matches = 0
    for i in range(min(len(reference_tokens), len(solution_tokens))):
        if reference_tokens[i] == solution_tokens[i]:
            matches += 1

    return matches / max_len


def longest_common_contiguous_length(a, b):
    """
    Longest common contiguous token block.
    """
    if not a or not b:
        return 0

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    best = 0

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])

    return best


def program_order_score(reference_tokens, solution_tokens):
    """
    Measures whether the relative order of tokens is preserved.
    Based on normalized Longest Common contiguous token block.
    """
    if not reference_tokens:
        return 0.0

    lccl = longest_common_contiguous_length(reference_tokens, solution_tokens)
    return lccl / len(reference_tokens)


def levenshtein_distance(a, b):
    """
    Computes token-level Levenshtein distance.
    """
    n = len(a)
    m = len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[n][m]


def program_edit_score(reference_tokens, solution_tokens):
    """
    Converts edit distance into a similarity score in [0, 1].
    1 means identical programs.
    0 means maximally different according to normalized edit distance.
    """
    max_len = max(len(reference_tokens), len(solution_tokens))

    if max_len == 0:
        return 0.0

    distance = levenshtein_distance(reference_tokens, solution_tokens)
    return 1.0 - (distance / max_len)


def partial_correctness(pos_score, op_score, order_score, edit_score):
    """
    Combined metric using equal weights.
    """
    return 0.25 * pos_score + 0.25 * op_score + 0.25 * order_score + 0.25 * edit_score


def is_present(value):
    return pd.notna(value) and str(value).strip().lower() not in {"", "none", "nan"}


def best_effort_program(row, has_partial_column):
    """
    The program to score a task's structural closeness with: the actual
    solution if solved, otherwise the best partial-match candidate the
    search kept (see deepcoder/search.py) if one exists, otherwise nothing.

    Without this, unsolved tasks always score 0 on every structural metric
    purely because there is no program at all to compare - not because the
    search didn't get close. best_partial_solution lets those tasks show
    actual partial credit instead of a uniform 0.
    """
    if row["solved"]:
        return row["solution"]
    if has_partial_column and is_present(row.get("best_partial_solution")):
        return row["best_partial_solution"]
    return None


def main():
    df = pd.read_csv(INPUT_FILE)
    has_partial_column = "best_partial_solution" in df.columns

    results = []

    for _, row in df.iterrows():
        reference_tokens = tokenize_program(row["reference"])
        solution_tokens = tokenize_program(row["solution"])

        acc = accuracy(reference_tokens, solution_tokens)
        op_score = program_operation_score(reference_tokens, solution_tokens)
        pos_score = program_position_score(reference_tokens, solution_tokens)
        order_score = program_order_score(reference_tokens, solution_tokens)
        edit_score = program_edit_score(reference_tokens, solution_tokens)

        partial = partial_correctness(
            pos_score=pos_score,
            op_score=op_score,
            order_score=order_score,
            edit_score=edit_score
        )

        best_effort = best_effort_program(row, has_partial_column)
        best_effort_tokens = tokenize_program(best_effort)
        be_op_score = program_operation_score(reference_tokens, best_effort_tokens)
        be_pos_score = program_position_score(reference_tokens, best_effort_tokens)
        be_order_score = program_order_score(reference_tokens, best_effort_tokens)
        be_edit_score = program_edit_score(reference_tokens, best_effort_tokens)
        be_partial = partial_correctness(
            pos_score=be_pos_score,
            op_score=be_op_score,
            order_score=be_order_score,
            edit_score=be_edit_score,
        )

        results.append({
            "reference": row["reference"],
            "solution": row["solution"],
            "solved": row["solved"],
            "accuracy": acc,
            "program_operation_score": op_score,
            "program_position_score": pos_score,
            "program_order_score": order_score,
            "program_edit_score": edit_score,
            "partial_correctness": partial,
            "best_effort_program": best_effort,
            "best_effort_operation_score": be_op_score,
            "best_effort_position_score": be_pos_score,
            "best_effort_order_score": be_order_score,
            "best_effort_edit_score": be_edit_score,
            "best_effort_partial_correctness": be_partial,
            "nb_steps": row["nb_steps"],
            "wall_ms": row["wall_ms"],
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Saved {len(out_df)} rows to {OUTPUT_FILE}")
    print()
    print("Summary:")
    print(out_df[[
        "accuracy",
        "program_operation_score",
        "program_position_score",
        "program_order_score",
        "program_edit_score",
        "partial_correctness",
        "best_effort_partial_correctness",
    ]].describe())

    print()
    print("Solved count:")
    print(out_df["solved"].value_counts())

    unsolved = out_df[~out_df["solved"]]
    if len(unsolved):
        print()
        print(f"Unsolved tasks: {len(unsolved)}")
        print(f"  with a best-partial candidate: {int(unsolved['best_effort_program'].notna().sum())}")
        print(f"  mean partial_correctness (old, always 0): {unsolved['partial_correctness'].mean():.4f}")
        print(f"  mean best_effort_partial_correctness (new): {unsolved['best_effort_partial_correctness'].mean():.4f}")


if __name__ == "__main__":
    main()