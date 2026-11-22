# 数据说明

本项目使用三个公开数据集。原始数据**不纳入版本库**，运行 `python scripts/download_data.py`
自动下载并写入本目录，同时生成 `manifest.json` 记录 SHA-256 校验值。

## 1. FGRC-SCD（主数据集）
- **内容**：中文电信网络诈骗风险短信，10 类（9 类诈骗 + "无风险"），含 `文本/风险类别/案件编号`。
- **来源**：HuggingFace `Abooooo/FGRC-SCD`（基于 CCF23-EVAL 任务 6 案件合成并筛选）。
- **许可证**：MIT。
- **文件**：`raw/fgrc_scd/FGRC-SCD-sms.zip`（SHA-256: `7ffe26db…b7b0`，完整值见 `manifest.json`）。
- **规模**：finetuning_initial 60,530 条 + eval_initial 61,863 条。

## 2. mudou_spam（真实中文短信，用于补充真实正常短信分布）
- **内容**：真实中文垃圾/正常短信（标签 1=垃圾，0=正常），`标签\t文本\t分词`。
- **来源**：GitHub `shaonianruntu/Spam-Message-Classification`（`mudou_spam.train/test`）。
- **许可证**：未在仓库注明，仅用于非商业课程研究并注明来源，原始数据不二次分发。
- **规模**：约 8,500 条（去重后）。

## 3. FBS_SMS_Dataset（真实伪基站垃圾短信，用于外部召回验证）
- **内容**：来自中国伪基站的真实垃圾短信，14 类（广告 AD / 诈骗 FR / 违法 IL 等），已匿名化。
- **来源**：GitHub `Cypher-Z/FBS_SMS_Dataset`，需引用论文
  Zhang et al., "Lies in the Air: Characterizing Fake-base-station Spam Ecosystem in China", ACM CCS 2020。
- **许可证**：未在仓库注明，使用需注明来源并引用论文。
- **规模**：约 13,000 条。

## 构造数据
- `processed/campus_test.csv`：40 条校园场景样例（20 正常通知 + 20 诈骗），基于公开反诈话术整理，
  非真实个人数据，仅用于定性评估与演示。
- `processed/obfuscation_test.csv`：1,500 条混淆变体（空格打散 / 全角数字 / 零宽字符 / 符号穿插），
  用于鲁棒性测试。

## 隐私说明
所有处理在本机完成。训练前将手机号、银行卡号、URL、金额等实体替换为掩码令牌（`<PHONE>` 等），
日志默认仅记录脱敏文本与 SHA-256 哈希，不存储原文。
