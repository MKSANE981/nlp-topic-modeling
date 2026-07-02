"""End-to-end NLP topic modeling pipeline."""

import argparse
import pandas as pd
from preprocess import preprocess_corpus
from lda_model import build_corpus, train_lda, coherence_score, tune_topics, print_topics
from analysis import compute_topic_shares, ols_wage_regression, top_terms_per_topic


def run(
    data_path: str,
    text_col: str = "body",
    wage_col: str = "log_wage",
    n_topics: int = 12,
    tune: bool = False,
    min_topics: int = 5,
    max_topics: int = 20,
):
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"{len(df):,} documents loaded.")

    print("Preprocessing ...")
    tokenized = preprocess_corpus(df[text_col].fillna("").tolist())

    dictionary, corpus = build_corpus(tokenized)
    print(f"Vocabulary: {len(dictionary)} terms | Corpus: {len(corpus)} documents")

    if tune:
        print(f"Tuning topics ({min_topics} → {max_topics}) ...")
        scores = tune_topics(corpus, dictionary, tokenized, min_topics, max_topics)
        best_k = int(scores.loc[scores["coherence"].idxmax(), "k"])
        print(f"\nBest k={best_k}")
        n_topics = best_k

    print(f"Training LDA with {n_topics} topics ...")
    model = train_lda(corpus, dictionary, n_topics)
    score = coherence_score(model, tokenized, dictionary)
    print(f"Coherence (c_v): {score:.4f}")

    print_topics(model)

    if wage_col in df.columns:
        print("\nRunning wage regression ...")
        topic_df = compute_topic_shares(model, corpus, n_topics)
        topic_cols = list(topic_df.columns)
        reg_df = pd.concat([df.reset_index(drop=True), topic_df], axis=1)
        ols_wage_regression(reg_df, topic_cols, wage_col)

    return model, dictionary


def main():
    parser = argparse.ArgumentParser(description="NLP Topic Modeling Pipeline")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--text_col", default="body")
    parser.add_argument("--wage_col", default="log_wage")
    parser.add_argument("--n_topics", type=int, default=12)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--min_topics", type=int, default=5)
    parser.add_argument("--max_topics", type=int, default=20)
    args = parser.parse_args()
    run(
        args.data_path, args.text_col, args.wage_col,
        args.n_topics, args.tune, args.min_topics, args.max_topics,
    )


if __name__ == "__main__":
    main()