"""全局配置：路径、常量、类别定义、正则。"""
from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TABLES_DIR = ARTIFACTS_DIR / "tables"

REPORT_DIR = PROJECT_ROOT / "report"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 类别定义（与 FGRC-SCD 数据集一致）
# ---------------------------------------------------------------------------
NO_RISK_LABEL = "无风险"

# 顺序固定的类别表，用于多分类模型输出与图表
FRAUD_CATEGORIES = [
    "冒充电商物流客服类",
    "虚假网络投资理财类",
    "虚假信用服务类",
    "虚假购物、服务类",
    "冒充公检法及政府机关类",
    "冒充领导、熟人类",
    "网络婚恋、交友类",
    "冒充军警购物类诈骗",
    "网黑案件",
]

ALL_CATEGORIES = [NO_RISK_LABEL] + FRAUD_CATEGORIES

# 风险类别 -> 简要风险描述（用于 Agent 解释）
CATEGORY_RISK_DESC = {
    "无风险": "未发现明显风险特征",
    "冒充电商物流客服类": "冒充电商/物流/银行客服，以退款、理赔、账户异常为由诱导操作",
    "虚假网络投资理财类": "以高收益理财、刷单返利、虚拟币投资为诱饵诱导转账",
    "虚假信用服务类": "以贷款、办卡、征信修复等虚假信用服务诱导缴费",
    "虚假购物、服务类": "以低价商品、代考代办等虚假服务诱导付款",
    "冒充公检法及政府机关类": "冒充公安、检察院、法院等机关，以涉嫌犯罪为由要求配合",
    "冒充领导、熟人类": "冒充领导、亲友、熟人，以急用钱等理由要求转账",
    "网络婚恋、交友类": "以婚恋交友为名建立信任后实施诱导投资/转账",
    "冒充军警购物类诈骗": "冒充军警等采购名义实施的购物诈骗",
    "网黑案件": "涉嫌网络黑灰产（洗钱、跑分等）相关话术",
}

# ---------------------------------------------------------------------------
# 数据列名
# ---------------------------------------------------------------------------
COL_TEXT = "text"
COL_LABEL = "label"          # 二分类标签：0=无风险, 1=风险
COL_CATEGORY = "category"    # 细分类别（含"无风险"）
COL_CASE_ID = "case_id"      # 案件编号，用于防泄漏切分

# ---------------------------------------------------------------------------
# 实体 / PII 正则
# ---------------------------------------------------------------------------
RE_PHONE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8})(?!\d)")
RE_BANK_CARD = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
RE_AMOUNT = re.compile(r"\d+(?:\.\d+)?\s*(?:元|万|块|k|K)")
_URL_DELIMS = r"\s\u3000\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\u3010\u3011<>\"'\u4e00-\u9fff\u3400-\u4dbf"
RE_URL = re.compile(r"(?:https?://|www\.)[^" + _URL_DELIMS + r"]+", re.IGNORECASE)
RE_EMAIL = re.compile(r"(?<![a-zA-Z0-9_.+\-])[\w.+-]+@[\w-]+\.[\w.-]+(?![a-zA-Z0-9_.\-])")
RE_QQ = re.compile(r"(?<![a-zA-Z0-9])(?:QQ|qq|企鹅|扣扣)[:：号]?\s*\d{5,11}(?!\d)")
RE_WECHAT = re.compile(r"(?<![a-zA-Z0-9])(?:微信|VX|vx|V信|微\s*信|wei\s*xin)[:：号]?\s*[a-zA-Z][a-zA-Z0-9_]{4,19}")
RE_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")

# 实体掩码令牌（用于特征工程，提升泛化与隐私保护）
MASK_PHONE = "<PHONE>"
MASK_CARD = "<CARD>"
MASK_AMOUNT = "<AMOUNT>"
MASK_URL = "<URL>"
MASK_EMAIL = "<EMAIL>"
MASK_QQ = "<QQ>"
MASK_WECHAT = "<WECHAT>"
MASK_ID = "<ID>"
MASK_NUM = "<NUM>"

# 掩码顺序（先掩码更长/更具体的实体，避免误伤）
MASK_RULES = [
    (RE_ID_CARD, MASK_ID),
    (RE_BANK_CARD, MASK_CARD),
    (RE_URL, MASK_URL),
    (RE_EMAIL, MASK_EMAIL),
    (RE_QQ, MASK_QQ),
    (RE_WECHAT, MASK_WECHAT),
    (RE_PHONE, MASK_PHONE),
    (RE_AMOUNT, MASK_AMOUNT),
]

# ---------------------------------------------------------------------------
# Agent 默认参数（训练脚本会用验证集调优后覆盖）
# ---------------------------------------------------------------------------
DEFAULT_THRESH_HIGH = 0.70
DEFAULT_THRESH_LOW = 0.35
DEFAULT_FUSION_ALPHA = 0.80   # final = alpha*ml + (1-alpha)*rule（设计值，消融见 evaluate.py）
RULE_OVERRIDE_THRESHOLD = 0.92  # 强规则命中时把分数抬高到的下限
MAX_TEXT_LEN = 2000

# 二分类正类标签文本（展示用）
POSITIVE_LABEL = "风险"
NEGATIVE_LABEL = "无风险"


def ensure_dirs() -> None:
    """创建项目所需目录。"""
    for d in (RAW_DIR, PROCESSED_DIR, DEMO_DIR, MODELS_DIR, FIGURES_DIR,
              TABLES_DIR, REPORT_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
