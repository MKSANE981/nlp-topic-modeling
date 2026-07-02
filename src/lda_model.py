"""LDA topic model training with coherence-based tuning."""

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


def build_corpus(tokenized_docs: List[List[str]]) -> Tuple[corpora.Dictionary, list]:
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    return dictionary, corpus


def train_lda(
    corpus: list,
    dictionary: corpora.Dictionary,
    n_topics: int,
    passes: int = 20,
    seed: int = 42,
) -> LdaModel:
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=passes,
        random_state=seed,
        alpha="auto",
        eta="auto",
    )


def coherence_score(model: LdaModel, tokenized_docs, dictionary) -> float:
    cm = CoherenceModel(
        model=model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    return cm.get_coherence()


def tune_topics(
    corpus,
    dictionary,
    tokenized_docs,
    min_topics: int = 5,
    max_topics: int = 20,
    step: int = 1,
) -> pd.DataFrame:
    results = []
    for k in range(min_topics, max_topics + 1, step):
        model = train_lda(corpus, dictionary, k)
        score = coherence_score(model, tokenized_docs, dictionary)
        print(f"k={k:3d}  coherence={score:.4f}")
        results.append({"k": k, "coherence": score})
    return pd.DataFrame(results)


def print_topics(model: LdaModel, n_words: int = 10) -> None:
    for topic_id, words in model.print_topics(num_words=n_words):
        print(f"Topic {topic_id}: {words}")


def plot_coherence(df: pd.DataFrame, save_path: str = None) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(df["k"], df["coherence"], marker="o")
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence (c_v)")
    plt.title("LDA coherence vs number of topics")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()