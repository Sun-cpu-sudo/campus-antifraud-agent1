# 校园反诈助手：可疑短信识别与处置建议 Agent

《机器学习》课程期末项目。围绕"学生收到可疑诈骗短信难辨真伪"这一真实需求，构建一个
**离线、可复现、可评估**的端到端 Agent 原型：输入一条短信，系统通过文本预处理、规则引擎
（关键词/URL/混淆检测）与机器学习模型（TF-IDF + 逻辑回归）融合，输出风险等级、判定依据与
处置建议，并对中低置信度样本触发人工复核。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据（约 30MB，自动校验 SHA-256）
python scripts/download_data.py

# 3. 数据清洗与切分
python scripts/prepare_data.py

# 4. 训练模型（约 3 分钟，生成 artifacts/models/）
python scripts/train.py

# 5. 实验评估与图表（生成 artifacts/figures/ 与 artifacts/tables/）
python scripts/evaluate.py

# 6. 启动网页原型
streamlit run app.py
```

命令行单条检测：

```bash
python cli.py predict --text "我是公安局民警，你银行卡涉嫌洗钱，请转账到安全账户验证。"
```

批量检测与评估：

```bash
python cli.py batch --input data/demo/campus_test.csv --output result.csv --label-col label
```

## 项目结构

```
campus-antifraud-agent/
├── app.py                    # Streamlit 网页原型（单条/批量/人工复核/模型说明）
├── cli.py                    # 命令行入口
├── requirements.txt
├── scripts/
│   ├── download_data.py      # 下载公开数据集并校验
│   ├── prepare_data.py       # 清洗、去重、切分、构造测试集
│   ├── train.py              # 特征构建、模型训练、阈值调优、保存产物
│   ├── evaluate.py           # 模型对比、消融、外部验证、鲁棒性、误差分析
│   ├── build_report.py       # 生成 Word/PDF 报告
│   └── screenshot_ui.py      # 网页截图（Playwright 驱动系统 Edge）
├── src/antifraud/
│   ├── config.py             # 全局配置与类别/正则定义
│   ├── preprocessing.py      # 归一化、PII 掩码、实体抽取
│   ├── features.py           # jieba 分词 + 词/字 TF-IDF 特征联合
│   ├── rules.py              # 关键词/URL/混淆规则引擎
│   ├── models.py             # 训练、阈值调优、产物读写
│   ├── evaluation.py         # 指标、图表、自助置信区间
│   ├── agent.py              # Agent 工作流（融合/决策/解释）
│   └── storage.py            # SQLite 日志与人工复核队列
├── tests/                    # pytest 单元测试
├── data/                     # raw/processed/demo（原始数据不入库，见 data/README.md）
├── artifacts/                # models/figures/tables
├── docs/reuse_and_licenses.md# 复用组件与许可证清单
└── report/                   # 报告文档与截图
```

## 核心方法

- **数据**：FGRC-SCD（MIT，生成式诈骗短信 10 类）+ mudou（真实短信）混合训练；FBS、校园构造集、
  混淆集做外部/鲁棒性验证。训练前按"案件编号"之外的样本级分层切分（因该数据集同一案件的
  诈骗版与无害对照版共享编号），并做文本去重与近重复泄漏核查。
- **特征**：jieba 词级(1-2 gram) 与字级(2-4 gram) TF-IDF 特征联合（约 14 万维）；PII 掩码。
- **模型**：逻辑回归（与朴素贝叶斯、线性 SVM 对比），`class_weight=balanced`，阈值扫描调优。
- **规则引擎**：9 类诈骗关键词 + URL 风险启发式 + 混淆检测，与模型分数加权融合（α=0.8），
  强风险模式托底。
- **Agent 工作流**：校验 → 预处理 → 规则 → 模型 → 融合 → 三级风险决策 → 解释与建议 →
  中低置信度转人工复核；SQLite 记录脱敏日志。

## 主要结果（详见报告）

| 评估 | 指标 |
|------|------|
| 测试集（混合数据） | F1=0.9955，AUC=0.9982 |
| 外部真实短信 mudou 留出 | F1=0.9374，AUC=0.9742 |
| 外部伪基站短信 FBS 召回 | 诈骗类 0.874 / 违法服务 0.758 |
| 校园场景集 | 召回 1.0，F1=0.9756 |
| 混淆鲁棒性（5 种） | 召回均 100% |
| 领域差距消融 | 加入真实数据后校园精确率 0.655→0.952 |

## 测试

```bash
python -m pytest
```

## 许可与引用

本项目代码以 MIT 协议发布（见 LICENSE）。所复用组件与数据集的许可证与来源见
`docs/reuse_and_licenses.md` 与 `data/README.md`。
