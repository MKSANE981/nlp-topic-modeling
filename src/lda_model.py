"""
LDA topic model training, coherence scoring and visualisation.

We use gensim's LDA implementation with alpha="auto" and eta="auto" —
this enables asymmetric priors learned from the data, which consistently
produces better-separated topics than fixed symmetric priors. The trade-off
is longer training time (roughly 2x), but it's worth it for research quality.

Choosing the right number of topics k is the hardest part. We use the
c_v coherence metric (Röder et al., 2015) because it correlates best with
human judgment of topic interpretability — better than perplexity, which
only measures held-out likelihood and often peaks at k values that produce
unintelligible topics.
"""

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


def build_corpus(tokenized_docs: List[List[str]]) -> Tuple[corpora.Dictionary, list]:
    """
    Build a gensim Dictionary and bag-of-words corpus from tokenised documents.

    filter_extremes removes:
    - Words appearing in fewer than 5 documents (too rare to form coherent topics)
    - Words appearing in more than 50% of documents (too common to be discriminative)

    These thresholds are conservative — you may want to tighten them for
    very large corpora where rare words are still statistically meaningful.

    Args:
        tokenized_docs: Output of preprocess.preprocess_corpus().

    Returns:
        Tuple of (Dictionary, list of bag-of-words vectors).
    """
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
    """
    Train an LDA model on the given corpus.

    passes=20 means the model sees the entire corpus 20 times. More passes
    generally improve convergence, but with diminishing returns after ~15.
    Setting random_state ensures reproducibility — important for a research
    paper where results need to be exactly replicable.

    Args:
        corpus: List of bag-of-words vectors (from build_corpus).
        dictionary: gensim Dictionary mapping word IDs to strings.
        n_topics: Number of topics to extract.
        passes: Number of full passes over the corpus during training.
        seed: Random seed for reproducibility.

    Returns:
        Trained LdaModel instance.
    """
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=passes,
        random_state=seed,
        # alpha="auto" learns an asymmetric Dirichlet prior — some topics
        # will be more prevalent than others, which is realistic
        alpha="auto",
        eta="auto",
    )


def coherence_score(model: LdaModel, tokenized_docs: List[List[str]], dictionary: corpora.Dictionary) -> float:
    """
    Compute the c_v coherence score for a trained LDA model.

    c_v measures how often the top words of each topic co-occur in the
    corpus. Higher is better — scores above 0.5 are generally considered
    interpretable; above 0.6 is good for domain-specific corpora.

    Args:
        model: Trained LdaModel.
        tokenized_docs: Original tokenised documents (needed for co-occurrence stats).
        dictionary: gensim Dictionary.

    Returns:
        Coherence score as a float.
    """
    cm = CoherenceModel(
        model=model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,  # avoid spawning child processes on Windows
    )
    return cm.get_coherence()


def tune_topics(
    corpus: list,
    dictionary: corpora.Dictionary,
    tokenized_docs: List[List[str]],
    min_topics: int = 5,
    max_topics: int = 20,
    step: int = 1,
) -> pd.DataFrame:
    """
    Train multiple LDA models and compare their coherence scores.

    This is a grid search over the number of topics. It's slow (one
    full training run per value of k), but it's the most reliable way
    to pick k. The best k is where coherence peaks before declining —
    further growth usually means topics start splitting into near-duplicates.

    Args:
        corpus: Bag-of-words corpus.
        dictionary: gensim Dictionary.
        tokenized_docs: Tokenised documents for coherence computation.
        min_topics: Smallest k to try.
        max_topics: Largest k to try.
        step: Increment between k values.

    Returns:
        DataFrame with columns 'k' and 'coherence', one row per k.
    """
    results = []
    for k in range(min_topics, max_topics + 1, step):
        model = train_lda(corpus, dictionary, k)
        score = coherence_score(model, tokenized_docs, dictionary)
        print(f"k={k:3d}  coherence={score:.4f}")
        results.append({"k": k, "coherence": score})
    return pd.DataFrame(results)


def print_topics(model: LdaModel, n_words: int = 10) -> None:
    """
    Print the top n words for each topic in a readable format.

    The words are listed with their probability weights. A good topic
    should be interpretable from its top 10 words — if it looks like
    a random word salad, try a different k or review your preprocessing.
    """
    for topic_id, words in model.print_topics(num_words=n_words):
        print(f"Topic {topic_id}: {words}")


def plot_coherence(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot coherence vs number of topics to help pick the best k visually.

    Look for the elbow — the point where coherence stops increasing
    significantly. If the curve is flat or noisy, try more passes
    during training or revisit your preprocessing.

    Args:
        df: Output of tune_topics() with columns 'k' and 'coherence'.
        save_path: If provided, save the figure instead of displaying it.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(df["k"], df["coherence"], marker="o", color="#2196F3")
    plt.xlabel("Number of topics (k)")
    plt.ylabel("Coherence score (c_v)")
    plt.title("LDA coherence vs number of topics")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()