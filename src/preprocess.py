"""
Text preprocessing — cleaning, tokenisation and lemmatisation.

Good preprocessing is often more impactful than model choice for topic
modeling. LDA works on bag-of-words, so keeping noisy tokens (URLs,
punctuation, very common words) directly hurts topic coherence.

We use NLTK's WordNetLemmatizer — lightweight, no GPU dependency, and
sufficient for most English NLP tasks. For large production corpora where
morphological accuracy matters more, spaCy can be swapped in.
"""

import re
import string
import nltk
from typing import List

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STOP = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def clean(text: str) -> str:
    """
    Remove noise from raw text before tokenisation.

    Steps: lowercase → strip URLs → strip HTML tags → remove punctuation
    → collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, min_len: int = 3) -> List[str]:
    """
    Tokenise and stem a cleaned text string.

    PorterStemmer is fully in-memory (no corpus files) and sufficient
    for topic modeling where the goal is grouping related word forms,
    not morphological accuracy. "running", "runner", "runs" all become
    "run", which improves topic coherence.

    Filters: stop words, non-alpha tokens, tokens shorter than min_len.
    """
    return [
        _STEMMER.stem(word)
        for word in text.split()
        if word.isalpha()
        and word not in _STOP
        and len(word) >= min_len
    ]


def preprocess_corpus(texts: List[str]) -> List[List[str]]:
    """
    Apply the full preprocessing pipeline to a list of documents.

    Each document becomes a list of stemmed tokens — the format gensim expects.
    """
    return [tokenize_and_lemmatize(clean(t)) for t in texts]
