# NLP Topic Modeling — Developer Skill Analysis

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
Python · gensim · NLTK · spaCy · scikit-learn · pandas · matplotlib · statsmodels
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline on sample data
python src/pipeline.py --data_path ./data/posts_sample.csv

# Tune number of topics
python src/lda_model.py --min_topics 5 --max_topics 20
```

## Project Structure

```
nlp-topic-modeling/
├── src/
│   ├── preprocess.py     # Text cleaning & tokenisation
│   ├── lda_model.py      # LDA training & coherence scoring
│   ├── analysis.py       # Econometric skill–wage analysis
│   └── pipeline.py       # End-to-end orchestration
├── notebooks/
│   └── exploration.ipynb
├── data/
│   └── posts_sample.csv  # 5k-row public sample
├── requirements.txt
└── README.md
```