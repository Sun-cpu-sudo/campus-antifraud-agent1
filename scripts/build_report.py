"""生成期末报告（.docx）并转为 PDF。

用法： python scripts/build_report.py
输出： report/校园反诈助手期末报告.docx 与 report/校园反诈助手期末报告.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

from src.antifraud.config import (
    MODELS_DIR, TABLES_DIR, FIGURES_DIR, REPORT_DIR, ensure_dirs,
)
from src.antifraud.models import load_metrics_json

# ---------------------------------------------------------------------------
# 读取实验数据
# ---------------------------------------------------------------------------
SUMMARY = load_metrics_json(TABLES_DIR / "summary.json")
TEST = SUMMARY["test_binary"]
COMPARE = SUMMARY["comparison_val"]
FEAT = SUMMARY["feature_ablation"]
FUSION = SUMMARY["fusion_ablation"]
EXT = SUMMARY["external"]
GAP = EXT["domain_gap_fgrc_only"]

CN = {"宋体": "宋体", "黑体": "黑体"}


def _fmt(x, n=4):
    return f"{x:.{n}f}"


# ---------------------------------------------------------------------------
# docx 辅助
# ---------------------------------------------------------------------------
def set_run(run, size=12, bold=False, font="宋体", color=None, underline=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.font.bold = bold
    if underline:
        run.font.underline = True
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
    sizes = {1: 16, 2: 14, 3: 12}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run(r, size=sizes.get(level, 12), bold=True, font="黑体")
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    set_run(r, size=9, bold=False, font="宋体")
    return p


def add_picture(doc, path, width_in=5.8, caption=""):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width_in))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            add_caption(doc, caption)


def add_table(doc, headers, rows, col_widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
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
            set_run(r, size=10, bold=False, font="宋体")
    return t


def _cover_field_table(doc, rows, label_w=1.7, value_w=3.6):
    """按期末报告封面模板生成字段表（两列，值列加下划线）。"""
    t = doc.add_table(rows=len(rows), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, (label, value) in enumerate(rows):
        lc = t.rows[i].cells[0]
        vc = t.rows[i].cells[1]
        lc.width = Inches(label_w)
        vc.width = Inches(value_w)
        lr = lc.paragraphs[0].add_run(label)
        set_run(lr, size=14, font="宋体")
        vp = vc.paragraphs[0]
        vp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        text = value if value else "　" * 6
        vr = vp.add_run(text)
        set_run(vr, size=14, font="宋体", underline=True)
    return t


def make_workflow_figure(path: Path):
    """用 matplotlib 生成 Agent 工作流示意图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    steps = [
        ("输入短信", "#e3f2fd"),
        ("校验与\n预处理", "#e8f5e9"),
        ("规则引擎\n关键词/URL/混淆", "#fff3e0"),
        ("ML 模型\nTF-IDF+逻辑回归", "#fce4ec"),
        ("分数融合\n与风险分级", "#ede7f6"),
        ("解释 + 处置\n建议 + 复核", "#e0f7fa"),
    ]
    n = len(steps)
    w, h = 1.35, 1.1
    xs = [0.15 + i * (w + 0.25) for i in range(n)]
    for i, (label, color) in enumerate(steps):
        x, y = xs[i], 0.65
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                             fc=color, ec="#555", lw=1.2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9)
        if i < n - 1:
            ar = FancyArrowPatch((x + w + 0.02, y + h / 2),
                                 (xs[i + 1] - 0.02, y + h / 2),
                                 arrowstyle="-|>", mutation_scale=16, color="#333")
            ax.add_patch(ar)
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 报告内容
# ---------------------------------------------------------------------------
def build() -> Path:
    ensure_dirs()
    make_workflow_figure(FIGURES_DIR / "workflow.png")

    doc = Document()
    # 页面默认字体
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ================= 封面（对齐课程模板） =================
    for _ in range(5):
        add_para(doc, "", space_after=0)
    add_para(doc, "“机器学习”课程期末报告", size=24, bold=True, font="黑体",
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    for _ in range(4):
        add_para(doc, "", space_after=0)

    # 题目（项目名称）
    _cover_field_table(doc, [("题  目", "校园反诈助手：可疑短信识别与处置建议 Agent")])
    add_para(doc, "", space_after=0)
    add_para(doc, "", space_after=0)

    # 其余字段（学院/专业/学号/姓名/组员及分工/课程教师/完成日期）
    _cover_field_table(doc, [
        ("学   院", "电子电气工程学院"),
        ("专   业", "计算机科学与技术"),
        ("学号", "028123069"),
        ("姓名", "孙骁"),
        ("组员及分工", "个人独立完成"),
        ("课程教师", "邱禧荷"),
        ("完成日期", "2026.11.22"),
    ])

    doc.add_page_break()

    # ================= 摘要 =================
    add_heading(doc, "摘  要", 1)
    add_para(doc,
        "针对在校学生频繁收到诈骗短信却难以辨别真伪的真实痛点，本项目设计并实现了一个离线、"
        "可复现、可评估的“校园反诈助手”Agent。系统以中文短信文本为输入，通过文本预处理、"
        "规则引擎（9 类诈骗关键词、URL 风险启发式、混淆检测）与机器学习模型（词/字 TF-IDF + "
        "逻辑回归）的加权融合，输出三级风险等级、判定依据与处置建议，并对中低置信度样本触发"
        "人工复核。实验使用 MIT 许可的生成式诈骗短信数据集 FGRC-SCD 与真实短信数据混合训练，"
        "在测试集上 F1 达 0.9955、AUC 达 0.9982；在外部真实短信上 F1 达 0.9374；对伪基站诈骗"
        "短信召回 0.874；校园场景集召回 1.0。领域差距消融实验证明，加入真实短信数据后校园场景"
        "精确率由 0.655 提升至 0.952，验证了数据混合策略的有效性。", first_indent=24)
    add_para(doc, "关键词：短信诈骗识别；文本分类；TF-IDF；逻辑回归；Agent",
             space_after=12)

    # ================= 一、需求与目标 =================
    add_heading(doc, "一、需求与目标", 1)
    add_heading(doc, "1.1 目标用户与痛点", 2)
    add_para(doc,
        "目标用户为在校大学生。大学生社会经验相对不足，且校园环境中有大量通过伪基站、群发短信"
        "实施的电信网络诈骗，常见类型包括冒充公检法、冒充客服退款、刷单返利、校园贷、中奖诈骗等。"
        "学生面对这类短信时难以判断真伪，轻则点击钓鱼链接、泄露验证码，重则向诈骗账户转账造成"
        "财产损失。", first_indent=24)
    add_para(doc,
        "现有做法（手机自带拦截、关键词黑名单）存在明显不足：黑名单规则僵硬，容易误伤正常"
        "通知（如银行动账、快递、验证码），又对“变体写法”的诈骗短信漏报。因此需要一个既能给出"
        "风险判断、又能解释依据并给出处置建议的助手。", first_indent=24)
    add_para(doc,
        "从诈骗生态看，FBS_SMS_Dataset 对伪基站短信的刻画将其分为广告营销、金融诈骗（冒充银行"
        "钓鱼）、违法服务等大类，而 FGRC-SCD 进一步把电信诈骗细分为冒充公检法、冒充客服、刷单"
        "返利投资、虚假贷款、冒充熟人等九类。这些短信普遍利用权威身份伪造、时间紧迫性与利益诱惑"
        "三种心理机制，学生一旦轻信，往往在短时间内完成转账，事后追回困难。", first_indent=24)
    add_heading(doc, "1.2 项目范围与成功标准", 2)
    add_para(doc,
        "项目范围限定为“短信文本的风险识别与处置建议”：系统只提供判断与建议，不自动阻断短信、"
        "不代替公安机关，对高风险结论保持人工确认。成功标准定义为：在测试集上 F1 ≥ 0.90；在外部"
        "真实短信上召回 ≥ 0.90；对常见规避手法保持鲁棒；中低置信度样本能正确触发人工复核。",
        first_indent=24)
    add_heading(doc, "1.3 典型使用场景", 2)
    add_para(doc,
        "以三条典型输入说明任务入口与结果：(1) 学生收到“我是公安局民警，你银行卡涉嫌洗钱，请把"
        "资金转到安全账户验证”的短信，Agent 应判定高风险，给出“不点击、不转账、通过官方渠道核实、"
        "必要时拨打 96110”的建议；(2) 收到“您的包裹已到菜鸟驿站，取件码 8-3-210”的快递通知，Agent "
        "应判定低风险、不打扰用户；(3) 收到“恭喜中奖，请先支付 88 元手续费领取”的短信，Agent 应识别"
        "为虚假中奖并提示勿付款。三类场景分别对应“高风险拦截、正常放行、中风险提示”三种处置，覆盖"
        "了系统的核心价值，也为后文实验的定性评估提供了锚点。", first_indent=24)

    # ================= 二、方案与系统设计 =================
    add_heading(doc, "二、方案与系统设计", 1)
    add_heading(doc, "2.1 Agent 工作流", 2)
    add_para(doc,
        "Agent 采用确定性的六步流水线（见图 1）：(1) 输入校验（空输入、超长截断）；(2) 预处理与"
        "实体抽取（全角转半角、去零宽字符、抽取 URL/手机号/银行卡号等）；(3) 规则引擎（关键词、"
        "URL 风险、混淆检测）；(4) 机器学习模型（TF-IDF + 逻辑回归给出风险概率与疑似类型）；"
        "(5) 分数融合与三级风险分级；(6) 解释、处置建议与人工复核。整个流程本地运行，不调用任何"
        "外部服务，不依赖大模型 API。", first_indent=24)
    add_picture(doc, FIGURES_DIR / "workflow.png", 6.2, "图 1  Agent 工作流")

    add_para(doc,
        "在失败与安全处理上：(1) 空输入、超长输入给出校验提示并截断；(2) 模型文件缺失时提示"
        "先训练、而非静默崩溃；(3) 高风险结论只给建议、不自动执行任何操作，涉资金动作一律要求"
        "用户自行通过官方渠道核实；(4) 中风险与模型-规则分歧样本进入人工复核队列；(5) 所有处理"
        "在本机完成，日志只记录脱敏文本与哈希，不存储原文。", first_indent=24)
    add_para(doc,
        "各步骤的输入输出边界明确：校验步骤接收原始短信，输出“可继续/拒绝”及截断文本；预处理输出"
        "清洗后的文本与实体列表；规则引擎输出规则分、命中类别与理由；模型输出风险概率与疑似类型分布；"
        "融合与决策输出最终风险分、三级等级、置信度与是否需复核；解释与建议输出面向用户的结构化结果，"
        "同时向 SQLite 写入一条脱敏日志。清晰的边界使每一步都可单独测试与替换。", first_indent=24)

    add_heading(doc, "2.2 模型、工具与数据的关系", 2)
    add_para(doc,
        "数据层：FGRC-SCD 提供大规模诈骗/无害文本，mudou 提供真实正常/垃圾短信以缓解生成数据与"
        "真实分布的差异；FBS、校园构造集、混淆集用于外部与鲁棒性验证。模型层：逻辑回归为主分类器，"
        "线性 SVM 为多分类器（识别 9 类诈骗类型）。规则层：关键词/URL/混淆启发式作为可解释的补充"
        "信号。三者通过加权融合连接，最终由决策层输出三级风险与建议。", first_indent=24)

    add_heading(doc, "2.3 关键设计选择", 2)
    add_para(doc,
        "（1）离线优先：核心能力由本地传统机器学习模型承担，不依赖大模型 API，保证可复现与隐私。"
        "（2）模型+规则融合：模型提供概率，规则提供可解释依据与安全托底，二者互补。"
        "（3）三级风险与人工复核：高/中/低三级而非简单二分类，中风险与模型-规则分歧样本转入人工"
        "复核，体现“人机协作”。（4）隐私设计：训练前对 PII 掩码，日志仅存脱敏文本与哈希。"
        "备选方案中，仅规则或仅模型的方案在消融中均弱于融合；引入大模型 API 生成建议虽可增强自然"
        "语言表达，但会引入外部依赖、成本与隐私风险，故本项目以规则模板代替，保证离线可复现。",
        first_indent=24)

    add_heading(doc, "2.4 组件与数据流", 2)
    add_para(doc,
        "系统由七个组件构成：normalize（归一化）、extract_entities（实体抽取）、rule_engine（规则）、"
        "ml_infer（模型推理）、fusion（融合）、advise（建议）与 storage（日志与复核队列）。数据流为单向"
        "管道：原始短信经 normalize 与 extract_entities 得到清洗文本与实体；清洗文本同时喂给规则引擎与"
        "模型；两者分数在 fusion 按 α=0.8 加权并做强规则托底；结果进入 advise 生成建议，同时把脱敏文本"
        "与哈希写入 SQLite。人工复核队列独立于主流程，复核结果可导出作为后续改进的反馈数据。",
        first_indent=24)

    # ================= 三、机器学习方法 =================
    add_heading(doc, "三、机器学习方法", 1)
    add_heading(doc, "3.1 数据处理", 2)
    add_para(doc,
        "主数据集 FGRC-SCD 为 MIT 许可的中文诈骗短信，共 122,393 条、10 个类别（9 类诈骗 + “无风险”），"
        "由 CCF23-EVAL 任务 6 的真实案件合成。检查发现该数据集同一“案件编号”在诈骗版与无害对照版"
        "之间共享，故不能按案件切分；因此采用全局文本去重后按类别分层的样本级切分。进一步用 3-gram "
        "Jaccard 核查近重复泄漏，99% 以上测试样本与训练集最大相似度低于 0.8，确认无模板泄漏。",
        first_indent=24)
    add_para(doc,
        "为缓解生成数据与真实短信的分布差异，引入真实短信数据集 mudou（约 8,500 条，切分为训练补充 "
        "6,834 条与外部留出 1,709 条）。混合后共 128,475 条，按 70/15/15 切分为训练 89,932 / 验证 "
        "19,271 / 测试 19,272 条，风险占比约 66.8%。数据清洗包括去空文本、按归一化文本去重、丢弃"
        "归一化后标签冲突的样本（约 10 条）。标签层面，mudou 的真实标注存在噪声（部分营销短信被标"
        "为正常），这一现象将在误差分析中专门讨论。", first_indent=24)
    add_para(doc,
        "数据探索（EDA）显示：类别分布不均衡——“虚假网络投资理财”约占 30%、“无风险”约占 33%，其余"
        "八类占比 0.7%~13%；文本长度中位数约 67 字、多数集中在 50~80 字；约两成样本含 URL、约一成含"
        "手机号。这些统计直接指导了特征设计（字符 n-gram 更适合短文本）与类别不平衡处理（对罕见类别"
        "在评估时单独报告而非只看宏平均）。", first_indent=24)

    add_heading(doc, "3.2 特征与表示", 2)
    add_para(doc,
        "文本经清洗后，将手机号、银行卡号、URL、金额、邮箱、QQ/微信号等实体替换为掩码令牌"
        "（如 <PHONE>、<URL>），既保护隐私又避免模型记忆具体号码。特征采用词级与字级 TF-IDF 的联合："
        "jieba 分词后取 1-2 元词袋，以及 2-4 元字符 n-gram（char_wb），共约 14 万维，sublinear TF "
        "与 min_df/max_df 过滤。之所以以字级特征为主，是因为中文没有天然词边界，分词错误会引入噪声，"
        "而字符 n-gram 无需分词即可覆盖任意未登录词，对变体与错别字更鲁棒；词级特征则补充了词义"
        "粒度信息，二者通过 FeatureUnion 横向拼接。", first_indent=24)
    add_para(doc,
        "在表示学习上，本项目未采用预训练词向量或大模型嵌入，而选择 TF-IDF 稀疏表示，理由有三：其一，"
        "短信文本短且词汇高度集中于少数诈骗话术，词袋/字袋即可充分表达；其二，TF-IDF 训练快、无需 GPU、"
        "完全离线可复现，契合课程“传统机器学习方法”的要求；其三，线性模型系数天然可解释，能直接输出"
        "关键特征词，支撑后文的判定解释。", first_indent=24)

    add_heading(doc, "3.3 模型与训练", 2)
    add_para(doc,
        "对比了多数类基线、多项式/互补朴素贝叶斯、逻辑回归与线性 SVM（见表 1）。逻辑回归与线性 SVM "
        "性能接近且最优，选择逻辑回归作为主分类器，理由是其概率输出可直接用于置信度、系数可解释"
        "（可提取关键特征词）、且为课程核心方法。由于正负样本不均衡（风险约 2/3），统一采用 "
        "class_weight='balanced' 并在验证集上扫描阈值。基线设置上，以多数类分类器为最弱基线、"
        "朴素贝叶斯为传统文本分类基线，用于检验逻辑回归是否确有增益。", first_indent=24)
    add_table(doc,
        ["模型", "验证集 F1", "AUC", "PR-AUC"],
        [["Dummy(多数类)", _fmt(COMPARE["Dummy(多数类)"]["f1"]), _fmt(COMPARE["Dummy(多数类)"]["roc_auc"], 3), _fmt(COMPARE["Dummy(多数类)"]["pr_auc"], 3)],
         ["MultinomialNB", _fmt(COMPARE["MultinomialNB"]["f1"]), _fmt(COMPARE["MultinomialNB"]["roc_auc"], 3), _fmt(COMPARE["MultinomialNB"]["pr_auc"], 3)],
         ["ComplementNB", _fmt(COMPARE["ComplementNB"]["f1"]), _fmt(COMPARE["ComplementNB"]["roc_auc"], 3), _fmt(COMPARE["ComplementNB"]["pr_auc"], 3)],
         ["LogisticRegression", _fmt(COMPARE["LogisticRegression"]["f1"]), _fmt(COMPARE["LogisticRegression"]["roc_auc"], 3), _fmt(COMPARE["LogisticRegression"]["pr_auc"], 3)],
         ["LinearSVC", _fmt(COMPARE["LinearSVC"]["f1"]), _fmt(COMPARE["LinearSVC"]["roc_auc"], 3), _fmt(COMPARE["LinearSVC"]["pr_auc"], 3)]])
    add_caption(doc, "表 1  模型对比（验证集）")
    add_para(doc,
        "除二分类风险检测外，还训练了一个多分类模型识别具体诈骗类型：以线性 SVM（OvR，"
        "class_weight='balanced'）在 10 个类别上训练，作为 Agent 解释“疑似类型”的辅助组件。其 "
        "macro-F1 为 0.86，其中高频类别（冒充客服、投资理财等）F1 超过 0.9，而“网黑案件”“网络婚恋"
        "交友”等罕见类别偏低，这一差异本身即为后续改进提供了方向。", first_indent=24)

    add_heading(doc, "3.4 规则融合", 2)
    add_para(doc,
        "规则引擎由三部分组成：9 类诈骗关键词库（每类命中越多、该类别分越高）、URL 风险启发式"
        "（裸 IP、可疑顶级域、短链服务、Punycode、非常规端口、路径敏感词等）与混淆检测（零宽字符、"
        "全半角数字混用、汉字被空格或符号穿插等）。规则能够输出可解释的判定依据（如“命中强风险"
        "模式：公安+洗钱”），这是纯概率模型难以直接提供的。", first_indent=24)
    add_para(doc,
        "规则引擎输出 0~1 的规则分，与模型概率按 final = 0.8·P_ml + 0.2·P_rule 融合；命中“安全账户+"
        "转账”“公安+洗钱”等强风险模式时托底到 0.92。α=0.8 为设计取值，消融实验（见 5.3）表明在域内"
        "数据上融合与纯模型几乎持平（F1 0.9955 vs 0.9954），规则的主要价值在于可解释性与分布外鲁棒性。",
        first_indent=24)

    add_heading(doc, "3.5 模型评估与校准", 2)
    add_para(doc,
        "评估同时报告准确率、精确率、召回率、F1、ROC-AUC 与 PR-AUC。由于类别不均衡且本任务更关注"
        "“不漏报诈骗”（漏报代价高于误报），PR-AUC 比 ROC-AUC 更能反映真实排序质量，故作为主要指标"
        "之一。阈值选择在验证集上扫描 0.05~0.95，最大化 F1 得到二分类阈值 0.37；风险等级则用双阈值"
        "（低 0.44 / 高 0.84）把不确定区间划为“中风险”触发人工复核，兼顾安全与可用性。", first_indent=24)

    # ================= 四、Skills 与组件复用 =================
    add_heading(doc, "四、Skills 与组件复用", 1)
    add_para(doc,
        "本项目复用的主要开源组件及许可证见表 2，完整清单见附录 A。其中核心为 scikit-learn（模型与"
        "特征）、jieba（分词）、Streamlit（网页）、openpyxl（导出）与 anthropics/skills 的 xlsx skill"
        "（导出设计参考）。", first_indent=24)
    add_table(doc,
        ["组件", "版本", "许可证", "用途"],
        [["scikit-learn", "1.9.1", "BSD-3", "TF-IDF、逻辑回归、SVM、指标"],
         ["jieba", "0.42.1", "MIT", "中文分词"],
         ["Streamlit", "1.67.0", "Apache-2.0", "网页原型"],
         ["openpyxl", "3.1.5", "MIT", "Excel 导出"],
         ["FGRC-SCD", "—", "MIT", "主训练/评估数据"],
         ["xlsx skill", "main", "Apache-2.0", "导出设计参考"]])
    add_caption(doc, "表 2  主要复用组件（完整清单见附录）")

    add_para(doc,
        "深挖复用组件 xlsx skill：该 skill（anthropics/skills，Apache-2.0）解决“以表格为主要输入/输出”"
        "的任务，输入为表格规格、输出为表格文件，依赖 openpyxl/pandas/markitdown 与 LibreOffice 重算。"
        "我阅读了其 SKILL.md 与 recalc.py，确认其核心机制（openpyxl 写公式不缓存值、需重算等）与限制。"
        "由于本项目的导出物是“检测结果报告”（数值已计算、无公式），我按其格式化建议（专业字体、列宽、"
        "颜色标注）用 openpyxl 直接写入结果、按风险等级填充红/黄/绿底色，并有意识地省略了公式重算环节；"
        "导出文件已在 Excel 中打开核验无误。", first_indent=24)
    add_para(doc,
        "对每个复用组件“是否适合本问题”都做了判断：jieba 适合中文分词但存在新词切分误差，故配合"
        "字符 n-gram 兜底；Streamlit 适合快速搭建演示原型；openpyxl 适合生成 Excel 报告。集成后的行为"
        "通过 21 项单元测试与人工核验（Excel 打开、网页截图、命令行样例）逐项确认，复用并非“拿来即用”，"
        "而是按需配置、改造并验证。", first_indent=24)

    # ================= 五、实验设计与结果 =================
    add_heading(doc, "五、实验设计与结果", 1)
    add_heading(doc, "5.1 测试集与指标", 2)
    add_para(doc,
        "测试集包含：混合域内测试集（19,272 条）、外部真实短信 mudou 留出（1,709 条）、外部伪基站"
        "垃圾短信 FBS（13,028 条，全为正类）、校园场景构造集（40 条）、混淆鲁棒性集（1,500 条）。"
        "指标按任务选择准确率、精确率、召回率、F1、ROC-AUC、PR-AUC 与混淆矩阵。", first_indent=24)
    add_para(doc,
        "选择指标的考量：准确率在类别不均衡时具有迷惑性（仅预测多数类即可得到约 0.67），故以 F1 与 "
        "PR-AUC 为主；外部 FBS 全为正类，只能用召回率衡量；混淆鲁棒性用召回率衡量其是否漏掉被伪装的"
        "正类；多分类用宏平均以反映罕见类别的表现，避免高频类别掩盖短板。", first_indent=24)

    add_heading(doc, "5.2 主结果", 2)
    add_para(doc,
        f"混合域内测试集上，融合模型 F1={_fmt(TEST['f1'])}、精确率 {_fmt(TEST['precision'])}、"
        f"召回率 {_fmt(TEST['recall'])}、AUC {_fmt(TEST['roc_auc'], 4)}、PR-AUC {_fmt(TEST['pr_auc'], 4)}；"
        f"混淆矩阵为 TN={SUMMARY['confusion_matrix'][0][0]}、FP={SUMMARY['confusion_matrix'][0][1]}、"
        f"FN={SUMMARY['confusion_matrix'][1][0]}、TP={SUMMARY['confusion_matrix'][1][1]}。"
        f"多分类（9 类诈骗 + 无风险）macro-F1 为 {_fmt(SUMMARY['multiclass_macro_f1'], 4)}。",
        first_indent=24)
    add_picture(doc, FIGURES_DIR / "confusion_matrix.png", 4.0, "图 2  混淆矩阵")
    add_picture(doc, FIGURES_DIR / "roc_pr.png", 5.8, "图 3  ROC 与 PR 曲线")

    add_heading(doc, "5.3 对照与消融", 2)
    add_para(doc,
        f"特征消融（验证集 F1）：仅词级 {_fmt(FEAT['词级'])}、仅字级 {_fmt(FEAT['字级'])}、词+字 "
        f"{_fmt(FEAT['词+字'])}，字符 n-gram 贡献最大。融合消融（测试集 F1）：仅规则 {_fmt(FUSION['仅规则'])}、"
        f"仅模型 {_fmt(FUSION['仅模型'])}、融合 {_fmt(FUSION['融合(α=0.8)'])}，说明模型为主、规则为补充。"
        f"阈值扫描（验证集）确定融合阈值 {_fmt(SUMMARY['threshold'], 2)}、风险等级双阈值 "
        f"低 {_fmt(SUMMARY.get('threshold_low', 0.44), 2)} / 高 {_fmt(SUMMARY.get('threshold_high', 0.84), 2)}。",
        first_indent=24)
    add_para(doc,
        "融合权重的敏感性实验显示：α 从 0.5 增大到 1.0 时，测试 F1 由 0.947 单调升至 0.996，说明模型"
        "分数是主信号；α=0.8 与 α=1.0 仅差约 0.0003，代价极小却保留了规则的解释与托底能力，这正是"
        "固定取 α=0.8 的依据。", first_indent=24)
    add_picture(doc, FIGURES_DIR / "ablation_feature.png", 5.0, "图 4  特征消融")
    add_picture(doc, FIGURES_DIR / "ablation_fusion.png", 5.0, "图 5  融合消融")

    add_heading(doc, "5.4 外部验证与领域差距", 2)
    add_para(doc,
        f"外部真实短信 mudou 留出集：F1 {_fmt(EXT['mudou']['f1'])}、精确率 {_fmt(EXT['mudou']['precision'])}、"
        f"召回率 {_fmt(EXT['mudou']['recall'])}、AUC {_fmt(EXT['mudou']['roc_auc'], 4)}，说明模型能较好泛化到"
        f"真实短信。伪基站短信 FBS 召回：诈骗类 {_fmt(EXT['fbs_recall']['诈骗类'], 3)}、违法服务 "
        f"{_fmt(EXT['fbs_recall']['违法服务'], 3)}、广告营销 {_fmt(EXT['fbs_recall']['广告营销'], 3)}。"
        f"校园场景集：召回 1.0、精确率 {_fmt(EXT['campus']['precision'], 3)}、F1 {_fmt(EXT['campus']['f1'])}。",
        first_indent=24)
    add_para(doc,
        "领域差距消融是关键证据：仅用生成数据训练时，模型在真实短信上精确率仅 "
        f"{_fmt(GAP['mudou_holdout']['precision'], 3)}（AUC {_fmt(GAP['mudou_holdout']['roc_auc'], 3)}），"
        "在校园场景集上精确率仅 "
        f"{_fmt(GAP['campus']['precision'], 3)}，即大量正常通知被误判为风险；加入真实短信数据后，"
        f"真实短信精确率升至 {_fmt(EXT['mudou']['precision'], 3)}、校园精确率升至 "
        f"{_fmt(EXT['campus']['precision'], 3)}，而召回基本不降。这直接证明了数据混合改进的价值。",
        first_indent=24)
    add_picture(doc, FIGURES_DIR / "fbs_recall.png", 5.0, "图 6  外部伪基站短信召回")

    add_heading(doc, "5.5 鲁棒性与失败案例", 2)
    add_para(doc,
        "对 300 条诈骗短信施加空格打散、全角数字、零宽字符、符号穿插四种规避手法，得益于预处理归一化，"
        "召回率均保持 100%。失败案例集中在：多分类对“网黑案件”“网络婚恋交友”等罕见类别的 F1 偏低；"
        f"域内测试仍有 {SUMMARY['test_false_negative']} 条漏报与 {SUMMARY['test_false_positive']} 条误报，"
        "误报主要出现在含“报名/登录/验证码”等词的正常通知上。例如一条“【教务处】四六级报名本周截止，"
        "请尽快登录系统完成”的正常通知综合分 0.398 被误判，原因是“报名/登录系统”在训练数据中与诈骗"
        "话术共现；而“招租银行卡刷流水”“相亲交友平台推荐”等罕见类别诈骗因样本少、得分偏低。这提示"
        "模型对罕见类别与部分正常通知措辞的区分能力仍有限。", first_indent=24)
    add_para(doc,
        "从混淆矩阵看，域内测试 19,272 条中漏报 51 条、误报 65 条，合计错误率约 0.6%。漏报样本多为"
        "罕见类别或变体写法，误报样本多为含金融、验证码词汇的正常通知。这一误差主要来自训练数据分布"
        "（生成式、罕见类别样本少），而非模型容量，因此改进优先级是补充数据，而非更换更大模型。",
        first_indent=24)
    add_picture(doc, FIGURES_DIR / "robustness.png", 5.0, "图 7  混淆鲁棒性")

    add_heading(doc, "5.6 完整运行演示", 2)
    add_para(doc, "图 8 为网页原型的完整运行截图：输入一条冒充公检法的诈骗短信后，系统给出高风险判定、"
                  "命中依据（强风险模式、关键词、关键特征词）与处置建议。", first_indent=24)
    add_picture(doc, REPORT_DIR / "ui_scam_result.png", 5.8, "图 8  网页原型运行演示（诈骗短信）")

    # ================= 六、讨论与改进 =================
    add_heading(doc, "六、讨论与改进", 1)
    add_para(doc,
        "真实使用价值：对明确诈骗话术的识别接近满分，且能解释依据，可作为学生端的“第一道提醒”，"
        "并把人从重复判断中解放出来；在效率上单条检测约几十毫秒，可支撑批量场景。风险与边界：系统"
        "仅提供建议、不自动阻断；对生成数据训练带来的域外风险已通过混合真实数据缓解，但仍需注意"
        "模型可能对未见过的正常通知误报，因此保留中风险转人工复核。局限：主数据为合成生成、标签"
        "存在少量噪声；罕见诈骗类别样本少；未覆盖图片、短链落地页等内容的深度取证。后续改进方向："
        "加入更多真实标注数据与主动学习、引入 URL 实时信誉源、对短链落地页做多模态取证、并用概率"
        "校准与不确定性估计进一步刻画模型何时“不可信”。", first_indent=24)
    add_para(doc,
        "对“生成数据能否支撑真实需求”的反思：领域差距实验给出了量化回答——纯生成数据训练会系统性"
        "高估真实场景表现（真实短信 AUC 仅 0.62、校园精确率仅 0.655），这提醒在评估 Agent 时必须以真实"
        "分布的数据作为最终判据，而不能只报生成数据上的高分。同时，Agent 的价值不仅在于单点准确率，更"
        "在于把“判断—解释—处置—复核”串成闭环，让机器与人的责任边界清晰：机器负责快速初筛与解释，"
        "人负责最终决策。", first_indent=24)

    # ================= 参考文献 =================
    add_heading(doc, "参考文献", 1)
    refs = [
        "[1] Almeida T A, Hidalgo J M G, Yamakami A. Contributions to the study of SMS spam filtering: new collection and results[C]//ACM DOCENG, 2011.",
        "[2] 数据集 FGRC-SCD. https://huggingface.co/datasets/Abooooo/FGRC-SCD (MIT License).",
        "[3] Zhang Y, Liu B, Lu C, et al. Lies in the Air: Characterizing Fake-base-station Spam Ecosystem in China[C]//ACM CCS, 2020.",
        "[4] Pedregosa F, et al. Scikit-learn: Machine Learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.",
        "[5] 结巴分词 jieba. https://github.com/fxsjy/jieba (MIT License).",
        "[6] Streamlit: A faster way to build and share data apps. https://streamlit.io (Apache-2.0).",
        "[7] anthropics/skills. https://github.com/anthropics/skills (Apache-2.0).",
        "[8] shaonianruntu/Spam-Message-Classification. https://github.com/shaonianruntu/Spam-Message-Classification.",
    ]
    for r in refs:
        add_para(doc, r, size=10.5, space_after=3)

    # ================= 附录 =================
    doc.add_page_break()
    add_heading(doc, "附录 A  复用组件与许可证清单", 1)
    add_para(doc, "完整清单见代码仓库 docs/reuse_and_licenses.md，下表为摘要。", size=10.5)
    add_table(doc, ["组件", "许可证", "来源"],
              [["scikit-learn 1.9.1", "BSD-3-Clause", "github.com/scikit-learn/scikit-learn"],
               ["jieba 0.42.1", "MIT", "github.com/fxsjy/jieba"],
               ["Streamlit 1.67.0", "Apache-2.0", "streamlit.io"],
               ["pandas/numpy/scipy/matplotlib/seaborn", "BSD", "对应官方仓库"],
               ["openpyxl 3.1.5", "MIT", "openpyxl.readthedocs.io"],
               ["FGRC-SCD", "MIT", "huggingface.co/datasets/Abooooo/FGRC-SCD"],
               ["mudou_spam", "未注明", "github.com/shaonianruntu/Spam-Message-Classification"],
               ["FBS_SMS_Dataset", "未注明(引用CCS'20)", "github.com/Cypher-Z/FBS_SMS_Dataset"],
               ["xlsx skill", "Apache-2.0", "github.com/anthropics/skills"]])

    add_heading(doc, "附录 B  运行方法与测试样例", 1)
    add_para(doc, "运行步骤：pip install -r requirements.txt → python scripts/download_data.py → "
                  "python scripts/prepare_data.py → python scripts/train.py → python scripts/evaluate.py "
                  "→ streamlit run app.py。", size=10.5, first_indent=24)
    add_para(doc, "单元测试：python -m pytest（21 项全部通过）。", size=10.5, first_indent=24)
    add_para(doc, "测试样例（命令行）：", size=10.5)
    add_para(doc, "python cli.py predict --text \"我是公安局民警，你银行卡涉嫌洗钱，请转账到安全账户验证。\"",
             size=10, font="Consolas")
    add_para(doc, "python cli.py batch --input data/demo/campus_test.csv --output result.csv --label-col label",
             size=10, font="Consolas")

    add_heading(doc, "附录 C  人工智能辅助声明", 1)
    add_para(doc, "本项目的资料检索、代码编写、调试与报告润色使用了生成式 AI 工具辅助，所有生成内容、"
                  "代码与实验结果均已在本机运行核验，作者能够解释全部设计与代码。", first_indent=24)

    out = REPORT_DIR / "028123069_孙骁_校园反诈助手.docx"
    doc.save(str(out))
    return out


def main() -> None:
    docx_path = build()
    print("已生成 docx:", docx_path)
    _to_pdf_word(docx_path)


def _to_pdf_word(docx_path: Path) -> None:
    """通过本机 Word 转换 PDF（需 pywin32，或可手动用 Word 另存为 PDF）。"""
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(str(docx_path))
        pdf_path = docx_path.with_suffix(".pdf")
        doc.SaveAs2(str(pdf_path), FileFormat=17)
        doc.Close(False)
        word.Quit()
        print("已生成 pdf:", pdf_path)
    except Exception as e:
        print(f"PDF 转换失败（可手动用 Word 将 docx 另存为 PDF）：{e}")


if __name__ == "__main__":
    main()
