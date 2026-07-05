"""
Econometric analysis — skill-wage regressions using LDA topic proportions.

Once topics are extracted, each document gets a topic distribution vector
(how much of the document belongs to each topic). We treat these proportions
as continuous "skill intensity" measures and regress them against log wages.

The OLS estimator uses HC3 heteroskedasticity-robust standard errors
(MacKinnon & White, 1985) because wage residuals are almost never
homoskedastic in practice — variances tend to increase with wage level.

This analysis was developed as part of a CREST research internship (2024)
and formed the basis of the econometric section of the final report.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from typing import List


def compute_topic_shares(
    lda_model,
    corpus: list,
    n_topics: int,
) -> pd.DataFrame:
    """
    Convert LDA output into a tabular topic-proportion matrix.

    For each document, get_document_topics() returns a sparse list of
    (topic_id, probability) tuples. We expand this into a dense matrix
    with one column per topic (including zero-probability topics) so it
    can be directly merged with the metadata DataFrame for regression.

    Args:
        lda_model: Trained LdaModel instance.
        corpus: Bag-of-words corpus (one vector per document).
        n_topics: Total number of topics in the model.

    Returns:
        DataFrame with shape (n_docs, n_topics), columns named topic_0 … topic_k.
    """
    rows = []
    for doc_bow in corpus:
        # minimum_probability=0.0 forces gensim to return all topics, not just
        # the ones with non-negligible probability
        topic_dist = dict(lda_model.get_document_topics(doc_bow, minimum_probability=0.0))
        rows.append({f"topic_{i}": topic_dist.get(i, 0.0) for i in range(n_topics)})
    return pd.DataFrame(rows)


def ols_wage_regression(df: pd.DataFrame, topic_cols: List[str], wage_col: str = "log_wage") -> dict:
    """
    Run an OLS regression of log wages on topic proportions.

    The formula includes experience and location fixed effects as controls.
    We use log wages (not levels) because log-linear models give coefficients
    that are directly interpretable as percentage wage effects.

    Interpretation: a coefficient of 0.12 on topic_3 means that a one-unit
    increase in topic_3 proportion is associated with a 12% higher wage,
    conditional on experience and location.

    Args:
        df: DataFrame containing topic proportions, wage, experience
            and location columns.
        topic_cols: List of topic column names (output of compute_topic_shares).
        wage_col: Name of the log wage column.

    Returns:
        Dict with the fitted model, R², coefficients and p-values.
    """
    formula = f"{wage_col} ~ " + " + ".join(topic_cols) + " + experience + C(location)"
    # HC3 standard errors are more conservative than HC1/HC2 in small samples
    model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    print(model.summary())
    return {
        "model": model,
        "r2": model.rsquared,
        "coef": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
    }


def plot_topic_evolution(df: pd.DataFrame, topic_cols: List[str], year_col: str = "year") -> None:
    """
    Plot how average topic proportions evolve over time.

    A rising line means that topic's share of Stack Overflow posts is
    growing — a proxy for increasing demand for that skill cluster.
    This was the core visualisation in the CREST report showing the
    rise of ML/AI topics from 2018 to 2023.

    Args:
        df: DataFrame with topic proportion columns and a year column.
        topic_cols: List of topic column names to plot.
        year_col: Name of the year column.
    """
    yearly = df.groupby(year_col)[topic_cols].mean()
    yearly.plot(figsize=(12, 5), title="Topic share evolution over time")
    plt.ylabel("Mean topic proportion")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.show()


def top_terms_per_topic(lda_model, topic_ids: List[int], n_words: int = 10) -> pd.DataFrame:
    """
    Extract the top n terms for a selection of topics into a readable DataFrame.

    Useful for building a manual skill taxonomy — you read the top terms
    and assign a human-readable label to each topic (e.g. topic_4 = "Web Dev",
    topic_7 = "ML/AI"). This labelling step is unavoidable in topic modeling;
    it requires domain expertise and can't be automated reliably.

    Args:
        lda_model: Trained LdaModel instance.
        topic_ids: List of topic indices to include.
        n_words: Number of top words per topic.

    Returns:
        DataFrame with columns 'topic' and 'top_terms'.
    """
    rows = []
    for tid in topic_ids:
        terms = lda_model.show_topic(tid, topn=n_words)
        rows.append({
            "topic": f"topic_{tid}",
            "top_terms": ", ".join(w for w, _ in terms),
        })
    return pd.DataFrame(rows)