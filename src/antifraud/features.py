"""特征工程：jieba 分词 + 词/字 n-gram TF-IDF 特征联合。"""
from __future__ import annotations

from typing import Any, List

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .preprocessing import preprocess_for_feature

# jieba 静默初始化（不打印构建词典日志）
jieba.setLogLevel(60)

_WORD_MAX_FEATURES = 60000
_CHAR_MAX_FEATURES = 80000


def tokenize_words(text: str) -> List[str]:
    """jieba 分词，返回 token 列表（过滤单字符标点与空白）。"""
    return [t for t in jieba.lcut(text) if t.strip()]


def build_features() -> FeatureUnion:
    """构建词级(1-2 gram) + 字级(2-4 gram) TF-IDF 特征联合器。"""
    word_vec = TfidfVectorizer(
        tokenizer=tokenize_words,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=_WORD_MAX_FEATURES,
        sublinear_tf=True,
        lowercase=False,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=3,
        max_df=0.95,
        max_features=_CHAR_MAX_FEATURES,
        sublinear_tf=True,
        lowercase=False,
    )
    return FeatureUnion([
        ("word", word_vec),
        ("char", char_vec),
    ])


def make_word_features() -> TfidfVectorizer:
    """仅词级特征（用于消融实验）。"""
    return TfidfVectorizer(
        tokenizer=tokenize_words,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=_WORD_MAX_FEATURES,
        sublinear_tf=True,
        lowercase=False,
    )


def make_char_features() -> TfidfVectorizer:
    """仅字级特征（用于消融实验）。"""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=3,
        max_df=0.95,
        max_features=_CHAR_MAX_FEATURES,
        sublinear_tf=True,
        lowercase=False,
    )


def preprocess_series(texts: Any) -> List[str]:
    """对可迭代文本批量预处理（清洗 + 掩码）。"""
    return [preprocess_for_feature(t) for t in texts]
