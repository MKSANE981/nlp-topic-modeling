# NLP Topic Modeling — Developer Skill Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Stack](https://img.shields.io/badge/Stack-gensim%20%7C%20NLTK%20%7C%20statsmodels-orange)

Automatic classification of technical skills from Stack Overflow posts using
Latent Dirichlet Allocation (LDA), with downstream econometric analysis of
skill–wage relationships.

Developed during a research internship at **CREST (ENSAI)**, 2024.
Awarded the **highest grade of the cohort** for methodological rigour and originality.

## Research Question

> How have the most in-demand software developer skills evolved on Stack Overflow
> between 2015 and 2023, and what is their impact on wages?

## Pipeline

```
Stack Overflow posts
        ↓
    Preprocessing
    (tokenisation, stopwords, lemmatisation)
        ↓
    LDA Topic Model
    (gensim, coherence-optimised k)
        ↓
    Skill Taxonomy
    (manual labelling of top-10 terms per topic)
        ↓
    Econometric Analysis
    (OLS: skill → wage, controlling for experience & location)
```

## Key Results

- Identified **12 stable skill clusters** (systems, web, ML/AI, data, mobile…)
- ML/AI cluster grew from **8% → 23%** of posts between 2018–2023
- OLS estimates: ML skills carry a **+12–18% wage premium** vs general dev skills (conditional)

## Tech Stack

```
Python · gensim · NLTK · scikit-learn · pandas · matplotlib · statsmodels
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline on sample data (80-row demo included)
python src/pipeline.py --data_path ./data/posts.csv

# Tune number of topics (data-driven k selection)
python src/lda_model.py --min_topics 2 --max_topics 10
```

## Pipeline Interconnections

Steps are coupled — the output of each stage constrains the next:

```
data/posts.csv  (or real Stack Overflow data)
        ↓
preprocess.py  →  clean_text() + tokenize_and_lemmatize()
  ├─ stop-word removal shapes the vocabulary seen by LDA
  ├─ filter_extremes(no_below=5, no_above=0.5) — tokens appearing in
  │  fewer than 5 docs or more than 50% of docs are dropped;
  │  these thresholds are hardcoded and not data-driven
  └─ resulting dictionary size directly affects topic coherence scores
        ↓
lda_model.py  →  train_lda(k=4)
  ├─ k=4 is set as a fixed default for the demo; tune_topics() exists
  │  and scans a range of k values by coherence — use it on real data
  ├─ CoherenceModel runs single-process (processes=1) for Windows
  │  compatibility; remove the flag on Linux/Mac for faster scoring
  └─ topic-word distributions are the input to the manual labelling step
        ↓
analysis.py  →  OLS skill → wage regression
  ├─ topic assignments from LDA become the skill proxy variables
  └─ coefficient validity depends on topic stability (check perplexity
     across multiple random seeds before interpreting coefficients)
```

## Implementation Notes

| Note | Detail |
|------|--------|
| **Stemmer, not lemmatizer** | The pipeline uses NLTK `PorterStemmer` instead of spaCy lemmatization. spaCy's `en_core_web_sm` loads PyTorch/thinc, which exhausts Windows virtual memory on constrained machines. PorterStemmer is pure-Python and memory-safe. On a Linux server with enough RAM, replace with spaCy for higher-quality lemmas. |
| **k=4 hardcoded in demo** | Topics were not tuned on the 80-row sample — the corpus is too small for coherence scores to be meaningful. Run `tune_topics(min_k=2, max_k=10)` on a real corpus (≥500 posts) to let the data select k. |
| **filter\_extremes thresholds** | `no_below=5` requires a term to appear in at least 5 documents. With only 80 rows, this is a strict filter. Adjust to `no_below=2` on the demo data or leave as-is for real data. |
| **Key results (CREST internship)** | The results (12 clusters, ML/AI growth 8%→23%) are from the original research on a multi-year Stack Overflow dataset, not from the public demo. |

## Project Structure

```
nlp-topic-modeling/
├── src/
│   ├── preprocess.py     # Text cleaning & tokenisation (PorterStemmer)
│   ├── lda_model.py      # LDA training, coherence scoring, topic tuning
│   ├── analysis.py       # Econometric skill-wage analysis
│   └── pipeline.py       # End-to-end orchestration
├── notebooks/
│   └── exploration.ipynb
├── data/
│   ├── posts.csv         # 80-row demo dataset (included)
│   └── generate_sample.py
├── requirements.txt
└── README.md
```