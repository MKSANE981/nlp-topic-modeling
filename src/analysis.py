"""Econometric analysis — OLS skill-wage regressions."""

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
    rows = []
    for doc_bow in corpus:
        topic_dist = dict(lda_model.get_document_topics(doc_bow, minimum_probability=0.0))
        rows.append({f"topic_{i}": topic_dist.get(i, 0.0) for i in range(n_topics)})
    return pd.DataFrame(rows)


def ols_wage_regression(df: pd.DataFrame, topic_cols: List[str], wage_col: str = "log_wage") -> dict:
    formula = f"{wage_col} ~ " + " + ".join(topic_cols) + " + experience + C(location)"
    model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    print(model.summary())
    return {
        "model": model,
        "r2": model.rsquared,
        "coef": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
    }


def plot_topic_evolution(df: pd.DataFrame, topic_cols: List[str], year_col: str = "year") -> None:
    yearly = df.groupby(year_col)[topic_cols].mean()
    yearly.plot(figsize=(12, 5), title="Topic share evolution over time")
    plt.ylabel("Mean topic proportion")
    plt.tight_layout()
    plt.show()


def top_terms_per_topic(lda_model, topic_ids: List[int], n_words: int = 10) -> pd.DataFrame:
    rows = []
    for tid in topic_ids:
        terms = lda_model.show_topic(tid, topn=n_words)
        rows.append({
            "topic": f"topic_{tid}",
            "top_terms": ", ".join(w for w, _ in terms),
        })
    return pd.DataFrame(rows)