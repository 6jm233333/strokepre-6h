# 脑卒中超早期预后预测统计建模

本项目用于基于结构化电子病历数据开展脑卒中患者超早期二分类预后预测。当前版本聚焦入院后 6 小时窗口，使用 MIMIC 派生队列作为内部建模数据、MC-MED 派生队列作为外部验证数据，完成特征工程、特征筛选、统计建模、机器学习建模、外部验证、校准曲线与 SHAP 可解释性分析。

## 原始数据来源

本目录中的 CSV 是面向建模整理后的派生特征表，不是 PhysioNet 原始文件

| 数据源 | 版本 | 访问地址 | 说明 |
| --- | --- | --- | --- |
| MIMIC-III Waveform Database Matched Subset | 1.0 | https://physionet.org/content/mimic3wdb-matched/1.0/ | MIMIC-III 临床数据库匹配的波形和数值记录子集，用于关联 ICU 监测数据与临床信息。 |
| MC-MED | 1.0.0 | https://physionet.org/content/mc-med/1.0.0/ | 急诊多模态数据集，包含 visits、labs、numerics、radiology reports 等文件，可用于构建外部验证队列。 |

预后 label 不应从波形、生命体征或实验室异常直接推断，而应由出院去向、死亡信息、病程结局或人工整理的病例级轨迹证据抽取。具体 prompt 见 `代码/脑卒中预后标签提取提示词.md`。

## 项目结构

```text
数据及其他—作品编号/
├── README.md
├── requirements.txt
├── .gitignore
├── 代码/
│   ├── Code.py
│   ├── README.md
│   └── 脑卒中预后标签提取提示词.md
├── 数据/
│   ├── README.md
│   ├── MIMIC_Yuhou_EHR_Prognosis_Tw_6h.csv
│   └── MC_MED_Yuhou_EHR_Prognosis_Tw_6h.csv
├── 图表/
├── 结果/
└── 文档/
    └── 项目说明.md
```

## 数据说明

当前数据集均为由原始 PhysioNet 数据整理得到的 6 小时观察窗口结构化 EHR 派生特征表。

| 文件 | 样本量 | 标签分布 | 用途 |
| --- | ---: | --- | --- |
| `MIMIC_Yuhou_EHR_Prognosis_Tw_6h.csv` | 1714 | `LABEL=0`: 1055；`LABEL=1`: 659 | 内部训练、交叉验证 |
| `MC_MED_Yuhou_EHR_Prognosis_Tw_6h.csv` | 1062 | `LABEL=0`: 726；`LABEL=1`: 336 | 外部验证 |

标签含义：

| 字段 | 含义 |
| --- | --- |
| `LABEL=0` | 预后良好，数据中对应 `Improved` |
| `LABEL=1` | 预后不良，MIMIC 中对应 `Worsened_or_Died`，MC-MED 中对应 `Worsened_or_Severe` |

## 建模流程

1. 读取 MIMIC 6h 数据。
2. 对实验室指标和生命体征做异常值缩尾、缺失值中位数填补。
3. 构造动态特征，包括极差、脉压、平均动脉压和休克指数。
4. 使用 Boruta 与 LASSO 进行特征筛选。
5. 生成基线特征表和特征分布图。
6. 建立 Logistic 回归、随机森林、XGBoost、LightGBM 模型。
7. 使用五折交叉验证评价内部性能。
8. 使用 MC-MED 数据进行外部验证。
9. 绘制 ROC、校准曲线和 SHAP 解释图。

## 快速开始

在项目根目录或 `代码/` 目录下运行脚本。若在 `代码/` 目录运行，请确保两个 CSV 文件可被脚本读取，或将脚本中的 `file_path` 修改为 `../数据/xxx.csv`。

```bash
pip install -r requirements.txt
python 代码/Code.py
```

## 主要依赖

见 `requirements.txt`。

## 结果文件

脚本会生成基线表、模型性能表、ROC 曲线、校准曲线和 SHAP 图。建议后续将输出统一保存到：

- `结果/`：CSV、模型评价表、统计结果。
- `图表/`：ROC、校准曲线、SHAP、特征筛选图。
