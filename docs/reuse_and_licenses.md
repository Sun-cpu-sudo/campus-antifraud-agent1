# 复用组件与许可证清单

本项目复用了以下开源组件。各组件均注明来源、版本、许可证与用途，并按原项目要求引用。

## 一、代码库 / 框架

| 组件 | 版本 | 许可证 | 用途 |
|------|------|--------|------|
| scikit-learn | 1.9.1 | BSD-3-Clause | TF-IDF 特征、逻辑回归、线性 SVM、朴素贝叶斯、指标与校准 |
| jieba | 0.42.1 | MIT | 中文分词（词级 TF-IDF 的 tokenizer） |
| pandas | 3.1.0 | BSD-3-Clause | 数据清洗、切分、结果表格 |
| numpy | 2.5.4 | BSD-3-Clause | 数值计算 |
| scipy | 1.18.1 | BSD-3-Clause | 稀疏矩阵 |
| matplotlib | 3.11.2 | PSF-based | 结果图表 |
| seaborn | 0.13.2 | BSD-3-Clause | 绘图样式 |
| Streamlit | 1.67.0 | Apache-2.0 | 网页原型界面 |
| openpyxl | 3.1.5 | MIT | 批量检测结果导出 Excel |
| python-docx | 1.2.0 | MIT | 报告文档生成 |
| pytest | 9.2.0 | MIT | 单元测试 |
| Playwright | ≥1.49 | Apache-2.0 | 网页界面截图（驱动系统 Edge） |

## 二、数据集

| 数据集 | 来源 | 许可证 | 用途 |
|--------|------|--------|------|
| FGRC-SCD | HuggingFace `Abooooo/FGRC-SCD` | MIT | 主训练/评估数据 |
| mudou_spam | GitHub `shaonianruntu/Spam-Message-Classification` | 未注明 | 补充真实正常短信分布 |
| FBS_SMS_Dataset | GitHub `Cypher-Z/FBS_SMS_Dataset`（引用 CCS'20 论文） | 未注明 | 外部真实垃圾短信召回验证 |

## 三、复用的 Agent Skill

### xlsx（anthropics/skills）
- **来源**：`https://github.com/anthropics/skills/tree/main/skills/xlsx`（主分支）。
- **许可证**：Apache-2.0（见该 skill 目录 `LICENSE.txt`）。
- **用途**：作为批量检测结果导出 Excel 报告的设计参考。

**深入分析**：

1. **解决什么子任务**：当表格文件（.xlsx/.csv/.tsv 等）是主要输入或输出时，完成电子表格的
   创建、读取、编辑、格式与图表制作；对含公式的表格提供强制重算（recalc）流程。
2. **输入输出**：输入为用户的表格规格（表头、公式、格式要求）；输出为符合规格的表格文件。
   其脚本 `recalc.py` 以 LibreOffice 重算公式并返回 `{status, total_formulas, total_errors, ...}` 的 JSON。
3. **依赖与权限**：`openpyxl`、`pandas`、`markitdown`（Python 包）以及 LibreOffice（`soffice`）
   用于公式重算；需能读写本地文件。
4. **限制**：openpyxl 写入公式时不缓存计算结果，需额外重算；`data_only=True` 读取后若保存会
   永久丢失公式；合并单元格只能写左上角；跨工作表引用带空格的表名需加引号等（详见其 SKILL.md）。
5. **行为核验**：我阅读了 SKILL.md 与 `recalc.py`，确认其依赖 openpyxl 的公式字符串写入机制。
   由于本项目的导出物是**检测结果报告**（风险分、类型等为已计算数值，不涉及公式），我按其
   "专业字体、列宽、颜色标注" 的格式化建议，用 openpyxl 直接写入计算好的结果并按风险等级
   填充颜色，省略了公式重算环节（这是有意为之的适配，因为报告不含公式）。生成的文件已在
   Excel 中打开核验：表头、风险等级颜色、数值均正确。

**集成适配**：在 `app.py` 的 `_excel_report()` 中实现——中文化表头、风险等级条件填充色
（高/中/低红黄绿）、数值列保留 4 位小数、列宽自适应，并复用其"输出前核验"原则。

## 四、人工智能辅助声明

本项目的资料检索、代码编写、调试与报告润色使用了生成式 AI 工具（opencode/DeepSeek 等）辅助，
所有生成内容、代码与实验结果均已在本机运行核验，作者能够解释全部设计与代码。
