"""文本预处理：归一化、清洗、PII 掩码、实体抽取。"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict

from .config import (
    MASK_RULES,
    MASK_URL,
    RE_URL,
    RE_PHONE,
    RE_BANK_CARD,
    RE_AMOUNT,
    RE_EMAIL,
    RE_QQ,
    RE_WECHAT,
    RE_ID_CARD,
)

# 零宽字符、不可见字符（常用于规避过滤）
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
# 全角 -> 半角
_FULLWIDTH_MAP = {i + 0xFEE0: i for i in range(0x21, 0x7F)}
_FULLWIDTH_MAP[0x3000] = 0x20  # 全角空格

_WS_RE = re.compile(r"\s+")
# 只保留中英数字与常见标点，其余替换为空格（- 用 \u002d 表示避免转义告警）
_KEEP_RE = re.compile(
    r"[^\u4e00-\u9fffA-Za-z0-9\u3001\u3002\uff01\uff1f\uff1b\uff1a"
    r"\u201c\u201d\u2018\u2019\uff08\uff09\u3010\u3011\u002d+@./:%]+"
)


def normalize_fullwidth(text: str) -> str:
    """全角字符转半角（数字、字母、符号）。"""
    return text.translate(_FULLWIDTH_MAP)


def strip_invisible(text: str) -> str:
    """去除零宽/不可见字符。"""
    return _ZERO_WIDTH.sub("", text)


def clean_text(text: str) -> str:
    """基础清洗：去不可见字符、全角转半角、压缩空白、去杂字符。"""
    if text is None:
        return ""
    text = strip_invisible(text)
    text = normalize_fullwidth(text)
    text = text.lower()
    text = _WS_RE.sub(" ", text)
    text = _KEEP_RE.sub(" ", text)
    return text.strip()


def mask_pii(text: str) -> str:
    """将敏感实体替换为掩码令牌（用于特征工程，提升泛化与隐私）。"""
    for pattern, token in MASK_RULES:
        text = pattern.sub(token, text)
    return text


def preprocess_for_feature(text: str) -> str:
    """特征工程前的完整预处理：清洗 + PII 掩码。"""
    return mask_pii(clean_text(text))


@dataclass
class Entities:
    urls: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    cards: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    qq: List[str] = field(default_factory=list)
    wechat: List[str] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "urls": self.urls,
            "phones": self.phones,
            "cards": self.cards,
            "amounts": self.amounts,
            "emails": self.emails,
            "qq": self.qq,
            "wechat": self.wechat,
            "ids": self.ids,
        }

    def has_sensitive(self) -> bool:
        return bool(self.urls or self.phones or self.cards or self.qq
                    or self.wechat or self.ids or self.emails)


def extract_entities(text: str) -> Entities:
    """从原始文本抽取实体（URL/电话/银行卡/金额/邮箱/QQ/微信/身份证）。"""
    e = Entities()
    e.urls = RE_URL.findall(text)
    e.phones = RE_PHONE.findall(text)
    e.cards = RE_BANK_CARD.findall(text)
    e.amounts = RE_AMOUNT.findall(text)
    e.emails = RE_EMAIL.findall(text)
    e.qq = RE_QQ.findall(text)
    e.wechat = RE_WECHAT.findall(text)
    e.ids = RE_ID_CARD.findall(text)
    return e


def display_mask(text: str) -> str:
    """对展示文本做部分脱敏（手机号中间打码、银行卡尾号保留、URL 域名保留）。"""
    text = RE_PHONE.sub(lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:], text)
    text = RE_ID_CARD.sub(lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:], text)
    text = RE_QQ.sub(lambda m: m.group(0)[:2] + "****", text)
    text = RE_WECHAT.sub(lambda m: m.group(0)[:2] + "****", text)
    return text
