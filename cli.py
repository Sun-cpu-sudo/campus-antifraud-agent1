"""命令行入口：单条检测 / 批量检测 / 查看结果摘要 / 启动网页。

用法：
  python cli.py predict --text "短信内容"
  python cli.py batch --input data.csv --output results.csv [--label-col label]
  python cli.py summary
  python cli.py serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.antifraud.agent import AntifraudAgent
from src.antifraud.config import MODELS_DIR, TABLES_DIR


def _load_agent():
    return AntifraudAgent.from_dir(MODELS_DIR)


def cmd_predict(args):
    agent = _load_agent()
    res = agent.analyze(args.text, sender=args.sender)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


def cmd_batch(args):
    import numpy as np
    import pandas as pd
    agent = _load_agent()
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    col = args.text_col if args.text_col in df.columns else "text"
    if col not in df.columns:
        print(f"未找到文本列（尝试 {col}），可用列：{list(df.columns)}")
        sys.exit(1)
    rows = []
    for _, r in df.iterrows():
        res = agent.analyze(str(r[col]))
        rows.append({
            "risk_level": res.risk_level,
            "score": res.final_score,
            "ml_prob": res.ml_prob,
            "ml_type": res.ml_type,
            "need_review": res.need_review,
        })
    out = df.copy()
    for k in ["risk_level", "score", "ml_prob", "ml_type", "need_review"]:
        out[k] = [r[k] for r in rows]
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已写入 {args.output}，共 {len(out)} 条")

    if args.label_col and args.label_col in out.columns:
        from sklearn.metrics import accuracy_score, f1_score
        y = (out[args.label_col].astype(str).isin(
            ["1", "风险", "spam", "垃圾"])).astype(int)
        pred = (out["score"] >= agent.binary_threshold).astype(int)
        print(f"  对比标签列 {args.label_col}：accuracy="
              f"{accuracy_score(y, pred):.4f} F1={f1_score(y, pred, zero_division=0):.4f}")


def cmd_summary(args):
    p = TABLES_DIR / "summary.json"
    if not p.exists():
        print("未找到结果摘要，请先运行 python scripts/train.py 与 scripts/evaluate.py")
        sys.exit(1)
    print(json.dumps(json.load(open(p, encoding="utf-8")),
                     ensure_ascii=False, indent=2))


def cmd_serve(args):
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main():
    parser = argparse.ArgumentParser(description="校园反诈助手")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("predict", help="单条检测")
    p.add_argument("--text", required=True)
    p.add_argument("--sender", default=None)
    p.set_defaults(func=cmd_predict)

    b = sub.add_parser("batch", help="批量检测")
    b.add_argument("--input", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--text-col", default="text")
    b.add_argument("--label-col", default=None)
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("summary", help="查看实验结果摘要")
    s.set_defaults(func=cmd_summary)

    sv = sub.add_parser("serve", help="启动 Streamlit 网页")
    sv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
