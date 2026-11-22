"""生成《提交说明与补充材料（备注）》文档（docx + PDF 由外部转换）。

用法： python scripts/build_notes.py
输出： report/028123069_孙骁_校园反诈助手_提交说明.docx
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

from src.antifraud.config import REPORT_DIR, ensure_dirs

STU_ID = "028123069"
STU_NAME = "孙骁"
PROJECT = "校园反诈助手：可疑短信识别与处置建议 Agent"
REPO = "https://github.com/Sun-cpu-sudo/campus-antifraud-agent1"
DATE = "2026 年 11 月 25 日"


def set_run(run, size=12, bold=False, font="宋体", color=None):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_para(doc, text="", size=12, bold=False, font="宋体", align=None,
             space_after=6, first_indent=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if first_indent:
        p.paragraph_format.first_line_indent = Pt(first_indent)
    if text:
        r = p.add_run(text)
        set_run(r, size=size, bold=bold, font=font)
    return p


def add_heading(doc, text, level=1):
    sizes = {1: 15, 2: 12.5}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run(r, size=sizes.get(level, 12), bold=True, font="黑体")
    return p


def add_table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        set_run(r, size=10, bold=True, font="黑体")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            set_run(r, size=9.5, font="宋体")
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    return t


def build() -> Path:
    ensure_dirs()
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    add_para(doc, "提交说明与补充材料（备注）", size=18, bold=True, font="黑体",
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_para(doc, "《机器学习》课程期末项目", size=12, font="宋体",
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)

    add_table(doc, ["项目", "内容"], [
        ["项目名称", PROJECT],
        ["学号 / 姓名", f"{STU_ID} / {STU_NAME}"],
        ["学院 / 专业", "电子电气工程学院 / 计算机科学与技术"],
        ["完成日期", DATE],
    ], widths=[1.6, 4.8])

    add_heading(doc, "一、提交材料清单", 1)
    add_para(doc, "按课程要求，本次提交包含以下四项：", first_indent=24)
    add_table(doc, ["序号", "提交项", "具体文件 / 链接"], [
        ["①", "报告 PDF", f"report/{STU_ID}_{STU_NAME}_校园反诈助手.pdf（本报告，含封面与附录）"],
        ["②", "代码仓库", f"{REPO}（公开，含 README、requirements.txt、运行步骤）"],
        ["③", "演示 / 可运行原型", "仓库内含完整代码与网页截图；运行 streamlit run app.py 即可演示"],
        ["④", "复用组件清单与许可证/来源说明", "见报告附录 A 及本文档第五节（完整版）"],
    ], widths=[0.6, 2.2, 3.8])

    add_heading(doc, "二、代码仓库与复现步骤", 1)
    add_para(doc, f"仓库地址：{REPO}", first_indent=24)
    add_para(doc, "环境：Python 3.10+；依赖见 requirements.txt。复现命令：", first_indent=24)
    for line in [
        "pip install -r requirements.txt",
        "python scripts/download_data.py      # 下载并校验公开数据集",
        "python scripts/prepare_data.py       # 清洗、去重、按类别分层切分",
        "python scripts/train.py              # 训练模型并保存 artifacts/models/",
        "python scripts/evaluate.py           # 生成指标与图表",
        "streamlit run app.py                 # 启动网页原型",
        "python -m pytest                     # 运行 21 项单元测试",
    ]:
        add_para(doc, line, size=10, font="Consolas", space_after=1)
    add_para(doc, "说明：仓库已包含训练好的模型（artifacts/models/），可直接运行 app.py 演示，"
                  "无需重新训练。", first_indent=24)

    add_heading(doc, "三、数据说明", 1)
    add_para(doc, "原始数据集体积较大且含真实短信文本，为保护隐私与遵守仓库规范，未纳入代码仓库，"
                  "可通过 download_data.py 自动下载并做 SHA-256 校验。涉及数据集：", first_indent=24)
    add_table(doc, ["数据集", "用途", "许可证"], [
        ["FGRC-SCD", "主训练/评估（中文诈骗短信 10 类）", "MIT"],
        ["mudou_spam", "补充真实正常短信分布", "仓库未注明"],
        ["FBS_SMS_Dataset", "外部真实垃圾短信召回验证", "仓库未注明（引用 CCS'20）"],
    ], widths=[1.8, 3.6, 1.6])
    add_para(doc, "如评阅需要直接获取数据，可另附云盘链接：________________（请按需填写）。",
             first_indent=24)

    add_heading(doc, "四、演示方式", 1)
    add_para(doc, "网页原型（Streamlit）：运行 streamlit run app.py，在“单条检测”输入短信即可看到"
                  "风险等级、判定依据与处置建议；报告图 8 为完整运行截图。命令行示例：", first_indent=24)
    add_para(doc, "python cli.py predict --text \"我是公安局民警，请转账到安全账户验证。\"",
             size=10, font="Consolas")

    add_heading(doc, "五、复用组件清单与许可证", 1)
    add_para(doc, "代码库/框架：", bold=True, space_after=3)
    add_table(doc, ["组件", "版本", "许可证", "用途"], [
        ["scikit-learn", "1.9.1", "BSD-3-Clause", "TF-IDF、逻辑回归、SVM、指标"],
        ["jieba", "0.42.1", "MIT", "中文分词"],
        ["pandas", "3.1.0", "BSD-3-Clause", "数据处理"],
        ["numpy", "2.5.4", "BSD-3-Clause", "数值计算"],
        ["scipy", "1.18.1", "BSD-3-Clause", "稀疏矩阵"],
        ["matplotlib", "3.11.2", "PSF-based", "图表"],
        ["seaborn", "0.13.2", "BSD-3-Clause", "绘图样式"],
        ["Streamlit", "1.67.0", "Apache-2.0", "网页原型"],
        ["openpyxl", "3.1.5", "MIT", "Excel 导出"],
        ["python-docx", "1.2.0", "MIT", "报告生成"],
        ["pytest", "9.2.0", "MIT", "单元测试"],
        ["Playwright", ">=1.49", "Apache-2.0", "网页截图"],
    ], widths=[1.6, 1.0, 1.6, 2.8])
    add_para(doc, "复用 Agent Skill：xlsx（anthropics/skills，Apache-2.0），作为批量检测结果导出 Excel 的"
                  "设计参考；已按其格式化建议用 openpyxl 实现，并核验导出结果。深入分析见报告第四节与"
                  "仓库 docs/reuse_and_licenses.md。", first_indent=24)

    add_heading(doc, "六、人工智能辅助声明", 1)
    add_para(doc, "本项目的资料检索、代码编写、调试与报告润色使用了生成式 AI 工具辅助；所有生成内容、"
                  "代码与实验结果均已在本机运行核验，作者能够解释全部设计与代码。", first_indent=24)

    out = REPORT_DIR / f"{STU_ID}_{STU_NAME}_校园反诈助手_提交说明.docx"
    doc.save(str(out))
    return out


if __name__ == "__main__":
    p = build()
    print("已生成:", p)
