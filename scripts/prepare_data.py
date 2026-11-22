"""数据清洗、去重、按案件编号切分，并构造校园/混淆/外部测试集。

输出到 data/processed/ 与 data/demo/。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.antifraud.config import (
    RAW_DIR, PROCESSED_DIR, DEMO_DIR, ensure_dirs,
    COL_TEXT, COL_LABEL, COL_CATEGORY, COL_CASE_ID,
    NO_RISK_LABEL, FRAUD_CATEGORIES, RANDOM_STATE,
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def _normalize_key(text: str) -> str:
    t = re.sub(r"\s+", "", str(text))
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t)


def load_fgrc() -> pd.DataFrame:
    """合并 finetuning_initial 与 eval_initial，返回 DataFrame。"""
    rows = []
    base = RAW_DIR / "fgrc_scd" / "message"
    for name in ["finetuning_initial.json", "eval_initial.json"]:
        with open(base / name, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            rows.append({
                COL_TEXT: str(r["文本"]).strip(),
                COL_CATEGORY: r["风险类别"],
                COL_CASE_ID: str(r.get("案件编号", "")),
            })
    df = pd.DataFrame(rows)
    df[COL_LABEL] = (df[COL_CATEGORY] != NO_RISK_LABEL).astype(int)
    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """按归一化文本去重，丢弃标签冲突样本。"""
    df = df.copy()
    df["_key"] = df[COL_TEXT].map(_normalize_key)
    # 删除空 key
    df = df[df["_key"] != ""]
    # 归一化文本 -> 标签集合，冲突则整组删除
    grp = df.groupby("_key")[COL_LABEL].nunique()
    conflict_keys = set(grp[grp > 1].index)
    df = df[~df["_key"].isin(conflict_keys)]
    df = df.drop_duplicates(subset="_key", keep="first")
    df = df.drop(columns="_key").reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame) -> dict:
    """按样本分层切分 train/val/test。

    说明：FGRC-SCD 的"案件编号"在"风险"与"无风险"两类之间是共享的
    （同一案件同时生成诈骗版本与无害对照版本），因此不能按案件编号切分，
    否则会破坏类别分布。这里在全局去重后按类别分层做样本级切分。
    """
    from sklearn.model_selection import train_test_split

    train, rest = train_test_split(
        df, test_size=1 - TRAIN_RATIO, stratify=df[COL_CATEGORY],
        random_state=RANDOM_STATE)
    val, test = train_test_split(
        rest, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=rest[COL_CATEGORY], random_state=RANDOM_STATE)
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def build_campus_set() -> pd.DataFrame:
    """构造校园场景测试集（基于公开反诈话术整理的样例，非真实个人数据）。"""
    normal = [
        "教务处通知：期末考试成绩已公布，请登录教务系统查询。",
        "【图书馆】您预约的座位已到时间，请及时签到，逾期将释放。",
        "【校园卡】消费提醒：您在食堂消费15.5元，余额120.3元。",
        "同学你好，明天上午9点在三教201开班会，请准时参加并携带学生证。",
        "【快递】您的包裹已到菜鸟驿站，取件码8-3-210，请凭码领取。",
        "【体测中心】请于本周五下午参加体质测试，地点：东区操场。",
        "【团委】今晚19:00在大学生活动中心举办迎新晚会，欢迎参加。",
        "你的验证码是3284，用于登录校园网，5分钟内有效，请勿泄露给他人。",
        "【实验室】本周组会推迟到15:00，地点不变，请相互转告。",
        "【宿管】本周六宿舍楼进行卫生检查，请同学们提前整理内务。",
        "【财务处】助学金已发放至你的校园卡，请注意查收。",
        "【就业中心】明晚19:00在报告厅举办简历辅导讲座，欢迎参加。",
        "【校医院】体检结果已出，请于工作时间到校医院领取报告单。",
        "【网信办】校园网将于今晚23:00-次日5:00维护升级，请提前保存数据。",
        "【保卫处】请同学们注意用电安全，严禁在宿舍使用大功率电器。",
        "【社团】动漫社纳新面试定于周六14:00在大学生活动中心，请按时到场。",
        "【教学办】四六级报名将于本周五截止，请未报名的同学尽快登录系统完成。",
        "【校友办】校友回访活动定于下周三，欢迎志愿者报名参加。",
        "【后勤】东区食堂二楼明日开放新窗口，欢迎同学们前往用餐。",
        "【学工部】请同学们及时完成本学期综合素质测评线上填报。",
    ]
    scam = [
        ("我是公安局民警，你名下银行卡涉嫌洗钱，请立即配合调查，把资金转到安全账户验证，否则将冻结你所有账户。", "冒充公检法及政府机关类"),
        ("【校园快递】您的包裹在运输途中损坏，请点击链接领取3倍理赔金：http://kuaidi-claim.top/lingqu", "冒充电商物流客服类"),
        ("同学你好，招聘校园兼职，日结300元，点赞刷单返利，垫付100返130，加QQ 123456 了解。", "虚假网络投资理财类"),
        ("无抵押校园贷，额度最高5万，秒到账，只需先交500元手续费，微信联系客服。", "虚假信用服务类"),
        ("恭喜您获得校园抽奖一等奖iPhone15，请先支付88元运费手续费领取，点击链接操作。", "虚假购物、服务类"),
        ("我是你辅导员张老师，这是我的新号码，我在外面急用钱，先转3000元到这个账户，回头还你。", "冒充领导、熟人类"),
        ("加个好友呗，我最近在做虚拟币投资，收益很高，可以带你一起赚钱。", "网络婚恋、交友类"),
        ("【教务处】你的学籍信息异常，请点击链接核实并缴纳保证金，否则将影响毕业。", "冒充电商物流客服类"),
        ("部队采购需要一批物资，先由你垫付货款，回头给你回扣，请尽快安排。", "冒充军警购物类诈骗"),
        ("招租银行卡刷流水，日入过千，安全无风险，加微信了解。", "网黑案件"),
        ("检察院通知：你涉嫌一起刑事案件，请下载屏幕共享软件配合资金清查。", "冒充公检法及政府机关类"),
        ("【银行】你的账户存在异常扣款，请立即登录链接核实并解冻。", "冒充电商物流客服类"),
        ("限时特价，最新款手机只要999元，厂家直销，先付款先发货，点击抢购。", "虚假购物、服务类"),
        ("在家轻松赚钱，投资1000返1200，稳赚不赔，导师带单，名额有限。", "虚假网络投资理财类"),
        ("你申请的贷款已通过初审，额度5万元，因征信问题需先缴纳保证金3000元。", "虚假信用服务类"),
        ("我是王院长，正在开会不方便接电话，你先帮我转一笔钱到这个账号。", "冒充领导、熟人类"),
        ("充值返现活动，充100返200，多充多得，机会难得，速来领取。", "虚假网络投资理财类"),
        ("您的银行卡被冻结，请拨打客服电话并提供验证码解冻。", "冒充电商物流客服类"),
        ("相亲交友平台推荐优质单身青年，添加好友可获取联系方式。", "网络婚恋、交友类"),
        ("专业代刷平台，出借银行卡账号可得佣金，日结，安全无忧。", "网黑案件"),
    ]
    rows = []
    for t in normal:
        rows.append({COL_TEXT: t, COL_CATEGORY: NO_RISK_LABEL, COL_LABEL: 0,
                     COL_CASE_ID: "campus_normal"})
    for t, cat in scam:
        rows.append({COL_TEXT: t, COL_CATEGORY: cat, COL_LABEL: 1,
                     COL_CASE_ID: "campus_scam"})
    return pd.DataFrame(rows)


def build_obfuscation_set(spam_texts, n: int = 300) -> pd.DataFrame:
    """从垃圾短信样本构造混淆测试集，衡量鲁棒性。"""
    import numpy as np
    rng = np.random.default_rng(RANDOM_STATE)
    sample = list(rng.choice(spam_texts, size=min(n, len(spam_texts)),
                             replace=False))
    rows = []
    for t in sample:
        rows.append({COL_TEXT: t, COL_CATEGORY: "原始", COL_LABEL: 1})
        # 打散空格
        spaced = "".join(c + " " if "\u4e00" <= c <= "\u9fff" else c for c in t)
        rows.append({COL_TEXT: spaced, COL_CATEGORY: "空格打散", COL_LABEL: 1})
        # 全角数字
        fw = t.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
        rows.append({COL_TEXT: fw, COL_CATEGORY: "全角数字", COL_LABEL: 1})
        # 零宽字符
        zw = "\u200b".join(t)
        rows.append({COL_TEXT: zw, COL_CATEGORY: "零宽字符", COL_LABEL: 1})
        # 符号穿插（"安·全·账·户"）
        sep = "".join(c + "·" if "\u4e00" <= c <= "\u9fff" else c for c in t)
        rows.append({COL_TEXT: sep, COL_CATEGORY: "符号穿插", COL_LABEL: 1})
    return pd.DataFrame(rows)


def load_mudou() -> pd.DataFrame:
    rows = []
    for name in ["mudou_spam.train", "mudou_spam.test"]:
        df = pd.read_csv(RAW_DIR / "mudou" / name, sep="\t", header=None,
                         encoding="utf-8", on_bad_lines="skip", quoting=3)
        df = df.dropna(subset=[1])
        df[1] = df[1].astype(str)
        for label, text in zip(df[0], df[1]):
            t = str(text).strip()
            if not t:
                continue
            rows.append({COL_TEXT: t, COL_LABEL: int(label),
                         COL_CATEGORY: NO_RISK_LABEL if int(label) == 0
                         else "其他垃圾短信", COL_CASE_ID: "mudou"})
    return pd.DataFrame(rows).drop_duplicates(subset=COL_TEXT)


def split_mudou(mudou: pd.DataFrame) -> tuple:
    """把真实短信 mudou 切分为训练补充(80%)与外部留出测试(20%)。"""
    from sklearn.model_selection import train_test_split
    tr, ho = train_test_split(mudou, test_size=0.2, stratify=mudou[COL_LABEL],
                              random_state=RANDOM_STATE)
    return tr.reset_index(drop=True), ho.reset_index(drop=True)


def load_fbs() -> pd.DataFrame:
    bucket = {
        "AD": "广告营销", "FR": "诈骗类", "IL": "违法服务",
    }
    rows = []
    for p in sorted((RAW_DIR / "fbs").glob("*")):
        name = p.name
        cat = name.split("_", 1)[0] if "_" in name else "Other"
        bucket_name = bucket.get(cat, "其他")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                # FBS 数据为分词+占位符格式：还原为连续文本
                t = t.replace("URL", "http://example.com").replace("DIGIT", "123")
                t = t.replace(" ", "")
                rows.append({COL_TEXT: t, COL_LABEL: 1,
                             COL_CATEGORY: bucket_name,
                             COL_CASE_ID: "fbs"})
    return pd.DataFrame(rows).drop_duplicates(subset=COL_TEXT)


def main() -> None:
    ensure_dirs()
    print("加载并清洗 FGRC-SCD ...")
    fgrc = dedup(load_fgrc())
    fgrc.to_csv(PROCESSED_DIR / "fgrc_only.csv", index=False,
                encoding="utf-8-sig")
    print(f"  FGRC 去重后 {len(fgrc)} 条")

    print("加载真实短信 mudou 并切分 ...")
    mudou = load_mudou()
    mudou_train, mudou_holdout = split_mudou(mudou)
    print(f"  mudou 训练补充 {len(mudou_train)} 条 / 外部留出 {len(mudou_holdout)} 条")

    # 合并训练数据（生成数据 + 真实短信）
    combined = pd.concat([fgrc, mudou_train], ignore_index=True)
    combined = combined.drop_duplicates(subset=COL_TEXT).reset_index(drop=True)
    print(f"  合并训练数据 {len(combined)} 条")

    splits = stratified_split(combined)
    for k in splits:
        print(f"  {k}: {len(splits[k])} 条, 风险占比 "
              f"{splits[k][COL_LABEL].mean():.3f}")

    for k, d in splits.items():
        out = d[[COL_TEXT, COL_LABEL, COL_CATEGORY, COL_CASE_ID]]
        out.to_csv(PROCESSED_DIR / f"{k}.csv", index=False, encoding="utf-8-sig")

    combined.to_csv(PROCESSED_DIR / "full.csv", index=False, encoding="utf-8-sig")

    print("构造校园场景测试集 ...")
    campus = build_campus_set()
    campus.to_csv(PROCESSED_DIR / "campus_test.csv", index=False,
                  encoding="utf-8-sig")
    campus.to_csv(DEMO_DIR / "campus_test.csv", index=False, encoding="utf-8-sig")
    print(f"  校园集 {len(campus)} 条")

    print("构造混淆测试集 ...")
    spam_texts = splits["test"][splits["test"][COL_LABEL] == 1][COL_TEXT].tolist()
    obf = build_obfuscation_set(spam_texts)
    obf.to_csv(PROCESSED_DIR / "obfuscation_test.csv", index=False,
               encoding="utf-8-sig")
    print(f"  混淆集 {len(obf)} 条（源自 {len(spam_texts)} 条垃圾短信）")

    print("准备外部真实数据集 ...")
    mudou_holdout.to_csv(PROCESSED_DIR / "mudou_holdout.csv", index=False,
                         encoding="utf-8-sig")
    print(f"  mudou 外部留出 {len(mudou_holdout)} 条")
    fbs = load_fbs()
    fbs.to_csv(PROCESSED_DIR / "fbs.csv", index=False, encoding="utf-8-sig")
    print(f"  fbs {len(fbs)} 条, 类别 {Counter(fbs[COL_CATEGORY])}")

    # 样例 JSON（供 App 演示）
    samples = {
        "normal": "【图书馆】您预约的座位已到时间，请及时签到。",
        "scam": "我是公安局民警，你名下银行卡涉嫌洗钱，请立即把资金转到安全账户验证，否则冻结账户。",
        "obfuscated": "恭喜您 获 得 校园抽 奖 一等奖，请点 击 http://zhongjiang.top/lingqu 领取，先支付88元手续费。",
    }
    with open(DEMO_DIR / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print("全部完成。")


if __name__ == "__main__":
    main()
