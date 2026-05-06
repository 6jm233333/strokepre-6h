# 代码目录说明

保存脑卒中超早期预后预测的统计建模代码与提示词文档

## 文件

| 文件 | 说明 |
| --- | --- |
| `Code.py` | 主分析脚本，包含数据预处理、特征工程、特征筛选、统计检验、模型训练、内部验证、外部验证、校准曲线和 SHAP 分析。 |
| `脑卒中预后标签提取提示词.md` | 从项目资料整理出的预后标签提取提示词模板，用于从病例文本、病程记录或出院信息中结构化提取二分类预后标签。 |

## 运行提醒

当前脚本中的数据路径使用相对文件名，例如：

```python
file_path = 'MIMIC_Yuhou_EHR_Prognosis_Tw_6h.csv'
```

如果从项目根目录运行，建议改成：

```python
file_path = '../数据/MIMIC_Yuhou_EHR_Prognosis_Tw_6h.csv'
```

外部验证数据同理可指向：

```python
file_path = '../数据/MC_MED_Yuhou_EHR_Prognosis_Tw_6h.csv'
```

