"""规则引擎：关键词类别规则、URL 风险启发式、混淆文本检测。

规则基于反诈领域的公开话术特征整理，作为机器学习模型的补充信号，
提供可解释的判定依据，并在高风险强模式命中时对模型分数做托底。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from .config import FRAUD_CATEGORIES

# ---------------------------------------------------------------------------
# 关键词规则：每个类别一组关键词，命中越多得分越高
# ---------------------------------------------------------------------------
_KEYWORD_RULES: Dict[str, List[str]] = {
    "冒充电商物流客服类": [
        "客服", "退款", "理赔", "订单异常", "快递丢失", "包裹", "账户异常",
        "注销会员", "解冻", "保证金", "验证码", "屏幕共享", "刷脸", "误操作",
        "扣费", "取消订单", "物流", "理赔金", "备用金",
    ],
    "虚假网络投资理财类": [
        "投资", "理财", "返利", "刷单", "兼职", "点赞任务", "佣金", "垫付",
        "虚拟币", "比特币", "数字货币", "涨停", "内部消息", "稳赚", "高收益",
        "带单", "导师", "荐股", "期货", "炒币", "充值返现", "做任务",
    ],
    "虚假信用服务类": [
        "贷款", "无抵押", "低息", "额度", "秒到账", "下款", "征信", "修复征信",
        "黑户", "代办", "信用卡提额", "套现", "砍头息", "放款", "先交费",
    ],
    "虚假购物、服务类": [
        "代考", "代办", "低价", "秒杀", "抢购", "免费领取", "0元", "付邮费",
        "清仓", "甩卖", "内部价", "厂家直销", "赠品", "中奖", "领取奖品",
    ],
    "冒充公检法及政府机关类": [
        "公安", "检察院", "法院", "警察", "警官", "通缉", "洗钱", "涉案",
        "逮捕", "拘留", "资金清查", "安全账户", "协查", "立案", "刑事",
        "冻结账户", "通缉令", "配合调查", "保密",
    ],
    "冒充领导、熟人类": [
        "我是领导", "我是老师", "换号", "新号码", "存一下", "急用钱", "借钱",
        "转账", "不方便接电话", "在忙", "帮忙", "领导", "老师", "熟人",
    ],
    "网络婚恋、交友类": [
        "交友", "婚恋", "网恋", "相亲", "加好友", "处对象", "谈恋爱", "老公",
        "老婆", "亲爱的", "投资带我", "见面礼", "红包",
    ],
    "冒充军警购物类诈骗": [
        "部队", "武警", "消防", "军警", "采购", "物资", "订餐", "工程款",
        "回扣", "先垫付", "供应商",
    ],
    "网黑案件": [
        "跑分", "洗钱", "黑产", "两卡", "断卡", "出借银行卡", "租卡", "卖卡",
        "接码", "薅羊毛", "刷流水",
    ],
}

# 强风险模式：命中即高风险（用于规则托底）。元组内关键词需同时出现。
_STRONG_PATTERNS: List[Tuple[str, ...]] = [
    ("安全账户", "转账"),
    ("公安", "洗钱"),
    ("通缉", "配合调查"),
    ("警察", "转账"),
    ("验证码", "转账"),
    ("屏幕共享", "银行卡"),
    ("刷单", "垫付"),
    ("贷款", "先交费"),
    ("中奖", "手续费"),
    ("检察院", "资金清查"),
]

# 通用风险词（单条即显著加分）
_GENERIC_RISK_WORDS = [
    "涉嫌", "违法", "冻结", "扣款", "逾期", "威胁", "否则", "限时",
    "立即点击", "马上操作", "不操作将", "紧急", "最后通知", "官方",
]

_SUSPICIOUS_TLDS = {
    "top", "xyz", "tk", "ml", "ga", "cf", "gq", "pw", "club", "click",
    "link", "work", "loan", "win", "vip", "icu", "monster", "quest",
    "cyou", "buzz", "site", "rest", "host",
}

_SHORTENER_DOMAINS = {
    "t.cn", "tb.cn", "suo.im", "dwz.cn", "url.cn", "baiku.cn", "m6t.cn",
    "goo.gl", "bit.ly", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
}

_IP_HOST_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PUNYCODE_RE = re.compile(r"^xn--")
_MANY_SUBDOMAINS_RE = re.compile(r"^(?:[a-z0-9-]+\.){3,}[a-z]{2,}$")
_LONG_HOST_RE = re.compile(r"^[a-z0-9-]{40,}\.")
# 圈注数字/字母（①-⑨、⓪、ⓐ-ⓩ、Ⓐ-Ⓩ）
_CIRCLED_RE = re.compile(r"[\u2460-\u2468\u24ea\u24d0-\u24e9\u24b6-\u24cf]")
# 全角数字
_FULLWIDTH_DIGIT_RE = re.compile(r"[\uff10-\uff19]")
_HALFWIDTH_DIGIT_RE = re.compile(r"[0-9]")


@dataclass
class RuleResult:
    score: float                 # 0..1 规则综合风险分
    categories: List[str]        # 命中的风险类别
    reasons: List[str]           # 可解释依据
    url_risk: Dict = field(default_factory=dict)
    obfuscation: Dict = field(default_factory=dict)
    strong_hit: bool = False


# ---------------------------------------------------------------------------
# URL 风险
# ---------------------------------------------------------------------------
def _url_risk(url: str) -> Dict:
    """对单个 URL 做启发式风险评估，返回 {'score', 'reasons'}。"""
    reasons: List[str] = []
    score = 0.0
    url = url.strip().rstrip(".,;:!?)]}>")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        p = urlparse(url)
    except ValueError:
        return {"score": 0.5, "reasons": ["URL 无法解析"]}

    host = (p.hostname or "").lower()
    try:
        port = p.port
    except ValueError:
        port = None
    scheme = p.scheme.lower()

    if not host:
        return {"score": 0.6, "reasons": ["缺少主机名"]}

    if _IP_HOST_RE.match(host):
        reasons.append("主机为裸 IP 地址")
        score += 0.4

    if _PUNYCODE_RE.match(host):
        reasons.append("使用 Punycode 国际化域名")
        score += 0.3

    if any(host == d or host.endswith("." + d) for d in _SHORTENER_DOMAINS):
        reasons.append("疑似短链接服务")
        score += 0.2

    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in _SUSPICIOUS_TLDS:
        reasons.append(f"可疑顶级域名 .{tld}")
        score += 0.25

    if _MANY_SUBDOMAINS_RE.match(host):
        reasons.append("子域层级过多")
        score += 0.15

    if _LONG_HOST_RE.match(host):
        reasons.append("主机名过长")
        score += 0.1

    if scheme == "http":
        reasons.append("使用非加密 HTTP 协议")
        score += 0.1

    if port and port not in (80, 443):
        reasons.append(f"使用非常规端口 {port}")
        score += 0.15

    if "@" in (p.netloc or ""):
        reasons.append("URL 含 @ 符号（混淆）")
        score += 0.3

    suspicious_path_words = ["login", "verify", "bank", "pay", "update",
                             "secure", "account", "password", "confirm",
                             "change", "zhifu", "yinhang", "denglu"]
    path = (p.path or "").lower()
    hit = [w for w in suspicious_path_words if w in path]
    if hit:
        reasons.append("路径含敏感词: " + "/".join(hit[:3]))
        score += 0.15

    return {"score": min(score, 1.0), "reasons": reasons}


def check_urls(urls: List[str]) -> Dict:
    """批量 URL 风险评估，返回 {'score', 'reasons', 'per_url'}。"""
    if not urls:
        return {"score": 0.0, "reasons": [], "per_url": []}
    per_url = []
    scores = []
    reasons: List[str] = []
    for u in urls:
        r = _url_risk(u)
        per_url.append({"url": u, **r})
        scores.append(r["score"])
        reasons.extend(r["reasons"])
    return {
        "score": max(scores),
        "reasons": _dedup(reasons),
        "per_url": per_url,
    }


# ---------------------------------------------------------------------------
# 混淆检测
# ---------------------------------------------------------------------------
def check_obfuscation(original: str) -> Dict:
    """检测常见规避手法，返回 {'flags', 'score'}。"""
    flags: List[str] = []
    score = 0.0

    # 零宽字符
    if re.search(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", original):
        flags.append("含零宽字符")
        score += 0.4

    # 汉字被空格/符号打散（如 "公 安 机 关"、"安·全·账·户"）
    spaced = re.findall(r"(?:[\u4e00-\u9fff]\s){2,}[\u4e00-\u9fff]", original)
    if spaced:
        flags.append("汉字被空格打散")
        score += 0.3
    dotted = re.findall(r"[\u4e00-\u9fff][·•\*\u00b7][\u4e00-\u9fff]", original)
    if len(dotted) >= 2:
        flags.append("汉字被符号穿插")
        score += 0.35

    # 数字混杂全角/半角
    if _FULLWIDTH_DIGIT_RE.search(original) and _HALFWIDTH_DIGIT_RE.search(original):
        flags.append("全半角数字混用")
        score += 0.2

    # 常见谐音/变体敏感词
    homophones = {
        "薇信": "微信", "威信": "微信", "支fu宝": "支付宝", "支付宝钱包": "支付宝",
        "公检法": "公检法", "安全账户": "安全账户",
    }
    for k, v in homophones.items():
        if k in original and k != v:
            flags.append(f"变体词「{k}」")
            score += 0.2
            break

    # 大量同形/异体字符（圈注字符）
    if _CIRCLED_RE.search(original):
        flags.append("含圈注字符")
        score += 0.3

    return {"flags": flags, "score": min(score, 1.0)}


# ---------------------------------------------------------------------------
# 规则引擎主函数
# ---------------------------------------------------------------------------
def _dedup(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def run_rules(original: str, cleaned: str, urls: List[str]) -> RuleResult:
    """对一条短信执行完整规则评估。"""
    text = cleaned
    reasons: List[str] = []
    cat_hits: Dict[str, int] = {}
    cat_score: Dict[str, float] = {}

    for cat, kws in _KEYWORD_RULES.items():
        hits = [k for k in kws if k in text]
        if hits:
            cat_hits[cat] = hits
            # 命中关键词越多，该类别分越高（封顶 1）
            cat_score[cat] = min(1.0, 0.3 + 0.15 * len(hits))

    # 通用风险词
    generic_hits = [w for w in _GENERIC_RISK_WORDS if w in text]
    generic_bonus = min(0.4, 0.1 * len(generic_hits))

    # URL 风险
    url_risk = check_urls(urls)

    # 混淆检测
    obf = check_obfuscation(original)

    # 类别分（取最高）+ 通用词 + URL + 混淆，加权求和后裁剪
    base_cat = max(cat_score.values()) if cat_score else 0.0
    rule_score = min(1.0, 0.55 * base_cat + 0.25 * url_risk["score"]
                     + 0.15 * obf["score"] + generic_bonus)

    # 强模式托底
    strong_hit = False
    for pat in _STRONG_PATTERNS:
        if all(p in text for p in pat):
            strong_hit = True
            reasons.append("命中强风险模式：" + "+".join(pat))
            rule_score = max(rule_score, 0.95)
            break

    # 组装解释
    for cat, hits in cat_hits.items():
        reasons.append(f"命中「{cat}」关键词：{'、'.join(hits[:5])}")
    if generic_hits:
        reasons.append("含风险诱导词：" + "、".join(generic_hits[:5]))
    reasons.extend(url_risk["reasons"])
    for f in obf["flags"]:
        reasons.append("检测到规避手法：" + f)

    return RuleResult(
        score=rule_score,
        categories=sorted(cat_score, key=cat_score.get, reverse=True),
        reasons=_dedup(reasons),
        url_risk=url_risk,
        obfuscation=obf,
        strong_hit=strong_hit,
    )
