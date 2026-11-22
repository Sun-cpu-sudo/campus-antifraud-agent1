"""校园反诈助手 Streamlit 网页原型。

运行： streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.antifraud.agent import AntifraudAgent, ModelNotFoundError
from src.antifraud.config import (
    MODELS_DIR, FIGURES_DIR, TABLES_DIR, DEMO_DIR,
    DEFAULT_FUSION_ALPHA, DEFAULT_THRESH_HIGH, DEFAULT_THRESH_LOW,
)
from src.antifraud.storage import Storage

st.set_page_config(page_title="校园反诈助手", page_icon="🛡", layout="wide")


@st.cache_resource
def load_agent():
    return AntifraudAgent.from_dir(MODELS_DIR)


@st.cache_resource
def load_storage():
    return Storage()


def risk_style(level: str) -> str:
    return {
        "高风险": "#d32f2f",
        "中风险": "#f9a825",
        "低风险": "#2e7d32",
    }.get(level, "#555")


def risk_badge(level: str) -> str:
    color = risk_style(level)
    return (f"<span style='background:{color};color:white;padding:4px 12px;"
            f"border-radius:12px;font-weight:600'>{level}</span>")


def render_result(res, sender=None):
    if not res.ok:
        st.error(res.error)
        return
    st.markdown("### 检测结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div style='font-size:0.85rem'>风险等级</div>{risk_badge(res.risk_level)}",
                unsafe_allow_html=True)
    c2.metric("综合风险分", f"{res.final_score:.3f}")
    c3.metric("模型概率", f"{res.ml_prob:.3f}")
    c4.metric("规则分", f"{res.rule_score:.3f}")
    st.progress(min(1.0, res.final_score))

    if res.ml_type != "无风险":
        st.info(f"疑似诈骗类型：**{res.ml_type}**")
    if res.need_review:
        st.warning("该样本置信度不足或模型与规则存在分歧，建议转人工复核。")

    st.markdown("**判定依据**")
    for r in res.reasons:
        st.markdown(f"- {r}")

    if res.entities:
        ent = {k: v for k, v in res.entities.items() if v}
        if ent:
            st.markdown("**提取到的实体**")
            st.write(ent)

    st.markdown("**处置建议**")
    for a in res.advice:
        st.markdown(f"- ✅ {a}")

    st.caption(f"脱敏文本：{res.text_masked}")
    return res


def _excel_report(df: pd.DataFrame) -> bytes:
    """生成格式化 Excel 报告（复用/适配开源 xlsx 组件的工作流）。"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook()
    ws = wb.active
    ws.title = "检测结果"
    headers = ["序号", "脱敏文本", "风险等级", "综合分", "模型概率",
               "规则分", "疑似类型", "建议复核"]
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="4472C4")
        c.alignment = Alignment(horizontal="center")
    fill = {"高风险": "FFC7CE", "中风险": "FFEB9C", "低风险": "C6EFCE"}
    for i, r in df.iterrows():
        ws.append([i + 1, r.get("text_masked", ""), r.get("risk_level", ""),
                   round(float(r.get("score", 0)), 4),
                   round(float(r.get("ml_prob", 0)), 4),
                   round(float(r.get("rule_score", 0)), 4),
                   r.get("ml_type", ""), "是" if r.get("need_review") else ""])
        lvl = r.get("risk_level", "")
        if lvl in fill:
            ws.cell(ws.max_row, 3).fill = PatternFill("solid", fgColor=fill[lvl])
    widths = [6, 48, 10, 10, 10, 10, 18, 10]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i)].width = w
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def tab_single(agent, storage):
    st.subheader("单条短信检测")
    with st.expander("示例", expanded=True):
        try:
            samples = json.load(open(DEMO_DIR / "samples.json", encoding="utf-8"))
        except Exception:
            samples = {}
        c = st.columns(3)
        if c[0].button("示例：正常通知"):
            st.session_state["input_text"] = samples.get("normal", "")
        if c[1].button("示例：诈骗短信"):
            st.session_state["input_text"] = samples.get("scam", "")
        if c[2].button("示例：混淆诈骗"):
            st.session_state["input_text"] = samples.get("obfuscated", "")

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    text = st.text_area("短信内容", key="input_text", height=120)
    sender = st.text_input("发信号码（可选）", "")
    store_raw = st.checkbox("保存原文用于复核（默认仅保存脱敏文本）", value=False)

    if st.button("开始检测", type="primary"):
        res = agent.analyze(text, sender=sender)
        render_result(res, sender)
        if res.ok:
            pid = storage.log_prediction(res.to_dict(), sender=sender,
                                         store_raw=store_raw)
            if res.need_review:
                st.session_state.setdefault("review_pid", pid)
                st.info("已进入人工复核队列。")


def tab_batch(agent, storage):
    st.subheader("批量检测")
    st.caption("上传 CSV（需含一列文本，列名可指定；可选标签列用于自动评估）")
    up = st.file_uploader("上传 CSV", type=["csv"])
    if up:
        df = pd.read_csv(up, encoding="utf-8-sig")
        text_col = st.selectbox("文本列", df.columns,
                                index=df.columns.get_loc("text")
                                if "text" in df.columns else 0)
        label_col = st.selectbox("标签列（可选，用于评估）",
                                 ["（无）"] + list(df.columns))
        if st.button("开始批量检测", type="primary"):
            rows = []
            prog = st.progress(0.0)
            texts = df[text_col].astype(str).tolist()
            for i, t in enumerate(texts):
                res = agent.analyze(t)
                rows.append({
                    "text_masked": res.text_masked, "risk_level": res.risk_level,
                    "score": res.final_score, "ml_prob": res.ml_prob,
                    "rule_score": res.rule_score, "ml_type": res.ml_type,
                    "need_review": res.need_review,
                })
                prog.progress((i + 1) / len(texts))
            out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
            st.dataframe(out[["text_masked", "risk_level", "score", "ml_type"]])
            csv = out.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载 CSV", csv, "batch_result.csv",
                               "text/csv")
            st.download_button("下载 Excel 报告", _excel_report(out),
                               "batch_result.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if label_col != "（无）":
                from sklearn.metrics import accuracy_score, f1_score
                y = (out[label_col].astype(str).isin(["1", "风险", "spam", "垃圾"])).astype(int)
                pred = (out["score"] >= agent.binary_threshold).astype(int)
                st.metric("准确率", f"{accuracy_score(y, pred):.3f}")
                st.metric("F1", f"{f1_score(y, pred, zero_division=0):.3f}")


def tab_review(agent, storage):
    st.subheader("人工复核队列")
    st.caption("仅显示脱敏文本；复核结果作为反馈，可导出用于后续改进。")
    pending = storage.pending_reviews()
    stats = storage.review_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("待复核", len(pending))
    c2.metric("已标记垃圾", stats["spam"])
    c3.metric("已标记正常", stats["ham"])
    if not pending:
        st.write("暂无待复核样本。")
        return
    for p in pending:
        with st.container(border=True):
            st.markdown(f"**{p['risk_level']}** · 综合分 {p['final_score']:.3f} · "
                        f"类型 {p['ml_type']}")
            st.write(p["text_masked"])
            with st.expander("判定依据"):
                for r in p["reasons"]:
                    st.markdown(f"- {r}")
            c = st.columns([1, 1, 1, 2])
            if c[0].button("确认为垃圾", key=f"spam_{p['id']}"):
                storage.add_review(p["id"], "spam")
                st.rerun()
            if c[1].button("误报（正常）", key=f"ham_{p['id']}"):
                storage.add_review(p["id"], "ham")
                st.rerun()
            note = c[3].text_input("备注", key=f"note_{p['id']}")
            if c[2].button("跳过", key=f"skip_{p['id']}"):
                storage.add_review(p["id"], "skip", note)
                st.rerun()


def tab_model(agent):
    st.subheader("模型与数据说明")
    cfg_path = MODELS_DIR / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        st.write("当前配置：", cfg)
    for fig in ["model_comparison.png", "confusion_matrix.png",
                "threshold_curve.png", "roc_pr.png"]:
        p = FIGURES_DIR / fig
        if p.exists():
            st.image(str(p), caption=fig, width="stretch")


def main():
    st.title("🛡 校园反诈助手")
    st.caption("可疑短信识别与处置建议 Agent（离线 · 本地处理 · 保护隐私）")

    try:
        agent = load_agent()
    except ModelNotFoundError as e:
        st.error(f"模型未训练：{e}")
        st.info("请先运行：python scripts/download_data.py && "
                "python scripts/prepare_data.py && python scripts/train.py")
        return
    storage = load_storage()

    with st.sidebar:
        st.header("设置")
        th = st.slider("高风险阈值", 0.5, 0.99, float(agent.threshold_high), 0.01)
        tl = st.slider("低风险阈值", 0.05, 0.49, float(agent.threshold_low), 0.01)
        al = st.slider("融合权重 α", 0.5, 1.0, float(agent.fusion_alpha), 0.05)
        agent.threshold_high = th
        agent.threshold_low = tl
        agent.fusion_alpha = al
        st.divider()
        st.caption(f"已检测 {storage.prediction_count()} 条")
        if st.button("清空本地日志"):
            storage.clear()
            st.success("已清空")
        st.divider()
        st.caption("隐私说明：所有处理均在本机完成，不上传内容；"
                   "默认仅记录脱敏文本与哈希。")

    tabs = st.tabs(["单条检测", "批量检测", "人工复核", "模型与数据"])
    with tabs[0]:
        tab_single(agent, storage)
    with tabs[1]:
        tab_batch(agent, storage)
    with tabs[2]:
        tab_review(agent, storage)
    with tabs[3]:
        tab_model(agent)


if __name__ == "__main__":
    main()
