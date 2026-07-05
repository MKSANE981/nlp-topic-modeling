"""
Text preprocessing — cleaning, tokenisation and lemmatisation.

Good preprocessing is often more impactful than model choice for topic
modeling. LDA works on bag-of-words, so keeping noisy tokens (URLs,
punctuation, very common words) directly hurts topic coherence.

We use spaCy for lemmatisation because it handles morphology better than
NLTK's WordNetLemmatizer — "running", "ran" and "runs" all become "run",
which reduces vocabulary size and improves topic quality.

The pipeline can be slow on large corpora (spaCy's NLP object has overhead
per document). For 100k+ documents, consider batching with nlp.pipe().
"""

import re
import string
import nltk
import spacy
from typing import List

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

_STOP = set(stopwords.words("english"))
_NLP = None  # lazy-loaded on first use to avoid startup cost if not needed


def _get_nlp():
    """Load the spaCy model once and reuse it across calls."""
    global _NLP
    if _NLP is None:
        # We disable parser and NER — we only need the tokenizer and lemmatizer,
        # and disabling unused components speeds things up significantly
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP


def clean(text: str) -> str:
    """
    Remove noise from raw text before tokenisation.

    The cleaning steps, in order:
    1. Lowercase everything so "Python" and "python" are the same token.
    2. Remove URLs — they contribute nothing to topic meaning.
    3. Strip HTML tags (common in Stack Overflow data).
    4. Remove punctuation — LDA doesn't need it.
    5. Collapse multiple spaces into one.

    Args:
        text: Raw document string.

    Returns:
        Cleaned string, lowercased and stripped.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, min_len: int = 3) -> List[str]:
    """
    Tokenise and lemmatise a cleaned text string.

    We filter out:
    - spaCy stop words (is_stop flag)
    - Punctuation tokens
    - Words in NLTK's English stop word list (broader coverage)
    - Tokens shorter than min_len characters (mostly articles and prepositions
      that slip through stop word lists)
    - Non-alphabetic tokens (numbers, code snippets mixed with words, etc.)

    Args:
        text: Pre-cleaned text (output of clean()).
        min_len: Minimum token length to keep. 3 is a safe floor —
            "AI" is an exception you'd handle with a custom keep-list.

    Returns:
        List of lemmatised, filtered tokens.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    return [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.lemma_ not in _STOP
        and len(token.lemma_) >= min_len
        and token.is_alpha
    ]


def preprocess_corpus(texts: List[str]) -> List[List[str]]:
    """
    Apply the full preprocessing pipeline to a list of documents.

    This is the main entry point used by lda_model.py and pipeline.py.
    Each document becomes a list of tokens (not a string) because that's
    the format gensim's Dictionary and corpus expect.

    Args:
        texts: List of raw document strings.

    Returns:
        List of token lists — one per document.
    """
    return [tokenize_and_lemmatize(clean(t)) for t in texts]