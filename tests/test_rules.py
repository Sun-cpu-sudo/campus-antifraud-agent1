"""规则引擎单元测试。"""
from src.antifraud.preprocessing import clean_text, extract_entities
from src.antifraud.rules import run_rules, check_obfuscation, _url_risk


def _rule(text):
    c = clean_text(text)
    e = extract_entities(text)
    return run_rules(text, c, e.urls)


def test_scam_strong_pattern():
    r = _rule("我是公安民警，请把钱转账到安全账户验证，否则冻结账户。")
    assert r.strong_hit
    assert r.score >= 0.9


def test_scam_keyword_category():
    r = _rule("招聘兼职，点赞刷单返利，垫付100返130。")
    assert "虚假网络投资理财类" in r.categories


def test_normal_low_score():
    r = _rule("【图书馆】您预约的座位已到时间，请及时签到。")
    assert r.score < 0.5
    assert not r.strong_hit


def test_url_ip_host():
    r = _url_risk("http://192.168.1.1/login")
    assert r["score"] >= 0.3
    assert any("IP" in x for x in r["reasons"])


def test_url_suspicious_tld():
    r = _url_risk("http://example.xyz/verify")
    assert r["score"] >= 0.2


def test_obfuscation_zero_width():
    r = check_obfuscation("转账\u200b到\u200b安全账户")
    assert r["score"] >= 0.3
    assert any("零宽" in f for f in r["flags"])


def test_obfuscation_symbol_insert():
    r = check_obfuscation("安·全·账·户")
    assert r["score"] >= 0.3
    assert any("符号穿插" in f for f in r["flags"])
