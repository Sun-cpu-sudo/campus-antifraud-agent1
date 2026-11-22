"""预处理与实体抽取单元测试。"""
from src.antifraud.preprocessing import (
    clean_text, mask_pii, normalize_fullwidth, strip_invisible,
    extract_entities, display_mask,
)


def test_normalize_fullwidth():
    assert normalize_fullwidth("ＡＢＣ１２３") == "ABC123"
    assert normalize_fullwidth("０１２３") == "0123"


def test_strip_invisible():
    assert strip_invisible("abc\u200bdef\u200c") == "abcdef"
    assert strip_invisible("a\ufeffb") == "ab"


def test_clean_text():
    assert clean_text("  Hello   世界  ") == "hello 世界"
    assert clean_text("支　付　宝") == "支 付 宝"


def test_mask_pii_phone():
    out = mask_pii("联系 13812345678 咨询")
    assert "<PHONE>" in out
    assert "13812345678" not in out


def test_mask_pii_url():
    out = mask_pii("点击 http://example.com/x 领取")
    assert "<URL>" in out


def test_extract_entities():
    e = extract_entities("拨打13812345678 或访问 http://a.com 卡号6222021234567890")
    assert e.phones
    assert e.urls
    assert e.cards


def test_display_mask_phone():
    out = display_mask("联系13812345678")
    assert "138****5678" in out
    assert "13812345678" not in out
