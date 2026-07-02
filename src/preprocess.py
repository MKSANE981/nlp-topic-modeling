"""Text preprocessing: cleaning, tokenisation, lemmatisation."""

import re
import string
import nltk
import spacy
from typing import List

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords

_STOP = set(stopwords.words("english"))
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, min_len: int = 3) -> List[str]:
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
    return [tokenize_and_lemmatize(clean(t)) for t in texts]