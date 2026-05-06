# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:11:05 2026

@author: 17207
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import os
import statsmodels.api as sm
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate
import IPython
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import (average_precision_score, accuracy_score,precision_score, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from boruta import BorutaPy
import scipy.stats as stats

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) 

#数据
file_path = 'MIMIC_Yuhou_EHR_Prognosis_Tw_6h.csv'
df = pd.read_csv(file_path)


# 异常值处理与变量构造
def preprocess_and_engineer(data):
    df_clean = data.copy()

    winsorize_targets = [c for c in df_clean.columns if any(k in c for k in ['Glucose','Bicarbonate','Chloride', 'Creatinine', 'Hematocrit','Hemoglobin','Platelet', 'Potassium','Potassium','Sodium','Urea_Nitrogen',
                        'WBC','DBP','HR','RR','SBP','SPO2','PP'])]
    for col in winsorize_targets:
        upper = df_clean[col].quantile(0.99)
        lower = df_clean[col].quantile(0.01)
        df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
        df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])

    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    
    range_vars = ['Glucose','Bicarbonate','Chloride', 'Creatinine', 'Hematocrit','Hemoglobin','Platelet', 'Potassium','Potassium','Sodium','Urea_Nitrogen',
                  'WBC','DBP','HR','RR','SBP','SPO2','PP']
    for var in range_vars:
        if f'{var}_max' in df_clean.columns and f'{var}_min' in df_clean.columns:
            df_clean[f'{var}_range'] = (df_clean[f'{var}_max'] - df_clean[f'{var}_min']).clip(lower=0)

    df_clean['PP_mean'] = df_clean['SBP_mean'] - df_clean['DBP_mean']
    df_clean['MAP_mean'] = (df_clean['SBP_mean'] + 2 * df_clean['DBP_mean']) / 3
    df_clean['Shock_Index'] = df_clean['HR_mean'] / df_clean['SBP_mean']
    return df_clean

df_processed = preprocess_and_engineer(df)

target_continuous_vars = ['AGE',                                                      
                   'Glucose_mean','Glucose_min','Glucose_max','Glucose_last', 'Glucose_range',
                   'Bicarbonate_mean','Bicarbonate_min','Bicarbonate_max','Bicarbonate_last','Bicarbonate_range',
                   'Chloride_mean','Chloride_min','Chloride_max','Chloride_last', 'Chloride_range',
                   'Creatinine_mean','Creatinine_min','Creatinine_max','Creatinine_last','Creatinine_range',
                   'Hematocrit_mean','Hematocrit_min','Hematocrit_max','Hematocrit_last','Hematocrit_range',
                   'Hemoglobin_mean','Hemoglobin_min','Hemoglobin_max','Hemoglobin_last','Hemoglobin_range',
                   'Platelet_mean','Platelet_min','Platelet_max','Platelet_last','Platelet_range',
                   'Potassium_mean','Potassium_min','Potassium_max','Potassium_last','Potassium_range',
                   'Sodium_mean','Sodium_min','Sodium_max','Sodium_last','Sodium_range',
                   'Urea_Nitrogen_mean','Urea_Nitrogen_min','Urea_Nitrogen_max','Urea_Nitrogen_last','Urea_Nitrogen_range',
                   'WBC_mean','WBC_min','WBC_max','WBC_last','WBC_range',
                   'DBP_mean','DBP_min','DBP_max','DBP_range',
                   'HR_mean','HR_min','HR_max','HR_range',
                   'RR_mean','RR_min','RR_max','RR_range',
                   'SBP_mean','SBP_min','SBP_max','SBP_range',
                   'SPO2_mean','SPO2_min','SPO2_max','SPO2_range',
                   'PP_mean','MAP_mean','Shock_Index'] 
categorical_vars = ['GENDER_NUM', 'Hypertension', 'Diabetes', 'Hyperlipidemia', 
                    'Ischemic_Heart_Disease', 'Chronic_Kidney_Disease', 'Atrial_Fibrillation', 'Heart_Failure']
label_col = 'LABEL'

table1_results = []
group_names = df_processed[label_col].unique()
group1 = df_processed[df_processed[label_col] == group_names[0]]
group2 = df_processed[df_processed[label_col] == group_names[1]]

all_features = target_continuous_vars + categorical_vars
X = df_processed[all_features].copy()
y = df_processed[label_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Boruta 特征筛选
scaler_b = StandardScaler()
X_train_scaled = X_train.copy()

X_train_scaled[target_continuous_vars] = scaler_b.fit_transform(X_train[target_continuous_vars])

rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
feat_selector = BorutaPy(rf_boruta, n_estimators='auto', random_state=42,alpha=0.01, verbose=0)

feat_selector.fit(X_train_scaled.values, y_train)

selected_features_boruta = np.array(all_features)[feat_selector.support_].tolist()
print(f"Boruta 筛选出 {len(selected_features_boruta)} 个关键特征：{selected_features_boruta}")


# LASSO 精选 
X_train_boruta = X_train_scaled.values[:, feat_selector.support_]

lasso_selector = LogisticRegressionCV(
    cv=5, 
    penalty='l1', 
    solver='liblinear', 
    class_weight='balanced', 
    scoring='roc_auc', 
    max_iter=1000, 
    random_state=42
)
lasso_selector.fit(X_train_boruta, y_train)

lasso_coefs = lasso_selector.coef_[0]
final_mask = lasso_coefs != 0

final_selected_features = np.array(selected_features_boruta)[final_mask].tolist()

print(f"LASSO 筛选完成，最终保留特征 ({len(final_selected_features)}个): {final_selected_features}")

print(f"原始特征总数: {len(all_features)}")
print(f"Boruta 筛选后: {len(selected_features_boruta)}")
print(f"LASSO 最终精选: {len(final_selected_features)}")

final_importance = pd.DataFrame({
    'Feature': final_selected_features,
    'Lasso_Coef': lasso_coefs[final_mask]
}).sort_values(by='Lasso_Coef', key=abs, ascending=False)
print(final_importance)

name_mapping = {
    'AGE': '年龄',
    'Glucose_mean': '血糖均值',
    'Urea_Nitrogen_last': '尿素氮末次值',
    'WBC_max': '白细胞最大值',
    'DBP_min': '舒张压最小值',
    'DBP_range': '舒张压极差',
    'HR_range': '心率极差',
    'RR_mean': '呼吸频率均值',
    'RR_min': '呼吸频率最小值',
    'RR_max': '呼吸频率最大值',
    'SBP_max': '收缩压最大值',
    'PP_mean': '脉压均值',
    'Shock_Index': '休克指数',
    'Shock_Index': '休克指数',
    'SPO2_mean':'外周血氧饱和度均值',
    'SPO2_min':'外周血氧饱和度最小值',
    'SPO2_max':'外周血氧饱和度最大值',
}

plot_df = final_importance.copy()
plot_df['Feature_CN'] = plot_df['Feature'].map(name_mapping).fillna(plot_df['Feature'])
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid", font='SimHei') 

colors = ['#1f77b4' if x > 0 else '#d62728' for x in plot_df['Lasso_Coef']]

ax = sns.barplot(data=plot_df, x='Lasso_Coef', y='Feature_CN', palette=colors)

plt.title('LASSO 回归系数分布', fontsize=16, pad=20)
plt.xlabel('系数大小', fontsize=12)
plt.ylabel('')
plt.axvline(x=0, color='black', linestyle='-', lw=1.5, alpha=0.5)
plt.tight_layout()
plt.savefig('LASSO_Coefficients.png', dpi=300, bbox_inches='tight')

#基线表格  
target_continuous_vars = ['AGE', 'Glucose_mean', 'Urea_Nitrogen_last', 'WBC_max', 'DBP_min', 'DBP_range', 'HR_range', 'RR_mean', 'RR_min', 'RR_max', 'SBP_max', 'SPO2_mean', 'SPO2_min', 'SPO2_max', 'PP_mean', 'Shock_Index']

categorical_vars = []
label_col = 'LABEL'

name_mapping = {
    'AGE': '年龄（岁）',
    'Glucose_mean': '血糖均值（mg/dL）',
    'Urea_Nitrogen_last': '尿素氮末次值（mg/dL）',
    'WBC_max': '白细胞最大值（K/µL）',
    'DBP_min': '舒张压最小值（mmHg）',
    'DBP_range': '舒张压极差（mmHg）',
    'HR_range': '心率极差（beats/min）',
    'RR_mean': '呼吸频率均值（breaths/min）',
    'RR_min': '呼吸频率最小值（breaths/min）',
    'RR_max': '呼吸频率最大值（breaths/min）',
    'SBP_max': '收缩压最大值（mmHg）',
    'PP_mean': '脉压均值（mmHg）',
    'Shock_Index': '休克指数',
    'SPO2_mean':'外周血氧饱和度均值（%）',
    'SPO2_min':'外周血氧饱和度最小值（%）',
    'SPO2_max':'外周血氧饱和度最大值（%）',
}
table1_results = []
group_names = df_processed[label_col].unique()
group1 = df_processed[df_processed[label_col] == group_names[0]]
group2 = df_processed[df_processed[label_col] == group_names[1]]

for var in target_continuous_vars:
    var_cn = name_mapping.get(var, var)
    mean1, std1 = group1[var].mean(), group1[var].std()
    mean2, std2 = group2[var].mean(), group2[var].std()
    t_stat, p_val = stats.ttest_ind(group1[var].dropna(), group2[var].dropna(), equal_var=False)
    
    table1_results.append({
        'Variable': var,
        '变量': var_cn,
        f'{group_names[0]} (N={len(group1)})': f"{mean1:.2f} ± {std1:.2f}",
        f'{group_names[1]} (N={len(group2)})': f"{mean2:.2f} ± {std2:.2f}",
        'P-value': f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    })

for var in categorical_vars:
    contingency_table = pd.crosstab(df[var], df[label_col])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    table1_results.append({
        'Variable': f"{var} (Overall P-value)",
        f'{group_names[0]} (N={len(group1)})': "",
        f'{group_names[1]} (N={len(group2)})': "",
        'P-value': f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    })
    for val in contingency_table.index:
        count1 = contingency_table.loc[val, group_names[0]]
        prop1 = (count1 / len(group1)) * 100
        count2 = contingency_table.loc[val, group_names[1]]
        prop2 = (count2 / len(group2)) * 100
        
        table1_results.append({
            'Variable': f"  - {val}",
            f'{group_names[0]} (N={len(group1)})': f"{count1} ({prop1:.1f}%)",
            f'{group_names[1]} (N={len(group2)})': f"{count2} ({prop2:.1f}%)",
            'P-value': ""
        })

table1_df = pd.DataFrame(table1_results)
print(table1_df.to_string(index=False))
table1_df.to_csv('6h_Table1_Baseline.csv', index=False, encoding='utf-8-sig')   

#箱线图
features_to_plot = list(name_mapping.keys())
plot_df = df.copy()
plot_df['预后情况'] = plot_df['LABEL'].map({1: '预后不良', 0: '预后良好'})
save_dir = '箱线图'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def fix_micro_label(text):
    return text.replace('µ', r'$\mu$')

for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=plot_df, 
        x='预后情况', 
        y=feature, 
        palette='Set2', 
        showmeans=True,
        width=0.5,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"7"}
    )
    
    cn_name = name_mapping[feature]
    ylabel_text = fix_micro_label(cn_name) 
    plt.xlabel('')
    plt.ylabel(ylabel_text, fontsize=12)
    sns.despine() 
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    file_name = f"{save_dir}/Boxplot_{feature}.png"
    plt.savefig(file_name, dpi=300)
    #plt.show()

# logistic回归 使用逐步回归，根据AIC BIC准则选择模型
X_train_sel = X_train[final_selected_features]
X_test_sel = X_test[final_selected_features]
selected_cont = [f for f in target_continuous_vars if f in final_selected_features]

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), selected_cont)],
    remainder='passthrough'
)

X_train_final = X_train[final_selected_features]
X_train_sm = sm.add_constant(X_train_final) # 加入截距

def stepwise_selection(X, y, criterion='aic'):
    initial_features = X.columns.tolist()
    best_features = []
    best_score = float('inf')
    
    while len(initial_features) > 0:
        remaining_features = [f for f in initial_features if f not in best_features]
        new_pval = pd.Series(index=remaining_features, dtype='float64')
        
        for feature in remaining_features:
            model = sm.Logit(y, sm.add_constant(X[best_features + [feature]])).fit(disp=0)
            new_pval[feature] = model.aic if criterion == 'aic' else model.bic
            
        best_candidate = new_pval.idxmin()
        
        if new_pval.min() < best_score:
            best_score = new_pval.min()
            best_features.append(best_candidate)
        else:
            break
            
    return best_features, best_score

aic_features, aic_val = stepwise_selection(X_train_sm, y_train, criterion='aic')
bic_features, bic_val = stepwise_selection(X_train_sm, y_train, criterion='bic')

print(f"\n>>> AIC 选出的最优特征: {aic_features}")
print(f">>> BIC 选出的最优特征: {bic_features}")

final_model_features = bic_features if 'const' in bic_features else ['const'] + bic_features
final_logit = sm.Logit(y_train, X_train_sm[final_model_features]).fit()

print(final_logit.summary())

results_summary = pd.DataFrame({
    'OR': np.exp(final_logit.params),
    '95% CI Lower': np.exp(final_logit.conf_int()[0]),
    '95% CI Upper': np.exp(final_logit.conf_int()[1]),
    'P-value': final_logit.pvalues
})
print(results_summary)
results_summary.to_csv('Logit_Results_Summary.csv', encoding='utf-8-sig')

# Optuna 贝叶斯优化三种模型  五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_pipelines = {}
best_params_dict = {}

def optimize_model(model_name, objective_func, n_trials=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_func, n_trials=n_trials)
    best_params_dict[model_name] = study.best_params
    return study.best_params

def obj_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'class_weight': 'balanced', 'random_state': 42
    }
    pipe = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(**params))])
    return cross_val_score(pipe, X_train_sel, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()

def obj_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.0),
        'eval_metric': 'logloss', 'random_state': 42
    }
    pipe = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(**params))])
    return cross_val_score(pipe, X_train_sel, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()

def obj_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'class_weight': 'balanced', 'verbosity': -1, 'random_state': 42
    }
    pipe = Pipeline([('prep', preprocessor), ('clf', LGBMClassifier(**params))])
    return cross_val_score(pipe, X_train_sel, y_train, cv=skf, scoring='roc_auc', n_jobs=-1).mean()

rf_p = optimize_model('RandomForest', obj_rf, n_trials=20)
xgb_p = optimize_model('XGBoost', obj_xgb, n_trials=20)
lgb_p = optimize_model('LightGBM', obj_lgb, n_trials=20)


# 全模型五折交叉验证
logit_features = ['AGE', 'Glucose_mean', 'Shock_Index', 'SBP_max', 'SPO2_max', 'Urea_Nitrogen_last', 'WBC_max']

models = {
    'RandomForest': RandomForestClassifier(**rf_p, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(**xgb_p, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(**lgb_p, class_weight='balanced', verbosity=-1, random_state=42),
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42) # 参数由之前优化所得
}

best_pipelines = {}

final_cv_results = []
skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = {
    'roc_auc': 'roc_auc',
    'auprc': 'average_precision',  # AUPRC 在 sklearn 中对应 average_precision
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'brier': 'neg_brier_score'     # 注意：这是负值，越大越好（即绝对值越小越好）
}

for name, clf in models.items():
    if name == 'LogisticRegression':
        X_cv = X[logit_features]
        logit_cont_cv = [f for f in logit_features if f in target_continuous_vars]
        prep_cv = ColumnTransformer(
            transformers=[('num', StandardScaler(), logit_cont_cv)],
            remainder='passthrough'
        )
    else:
        X_cv = X[final_selected_features]
        prep_cv = preprocessor

    final_pipe = Pipeline([('prep', prep_cv), ('clf', clf)])
    best_pipelines[name] = final_pipe
    
    cv_metrics = cross_validate(
        final_pipe, X_cv, y, 
        cv=skf_final, 
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=False
    )
    
    brier_mean = -cv_metrics['test_brier'].mean() 
    brier_std = cv_metrics['test_brier'].std()

    final_cv_results.append({
        'Model': name,
        'AUC': f"{cv_metrics['test_roc_auc'].mean():.3f} ± {cv_metrics['test_roc_auc'].std():.3f}",
        'AUPRC': f"{cv_metrics['test_auprc'].mean():.3f} ± {cv_metrics['test_auprc'].std():.3f}",
        'Brier': f"{brier_mean:.3f} ± {brier_std:.3f}",
        'Accuracy': f"{cv_metrics['test_accuracy'].mean():.3f} ± {cv_metrics['test_accuracy'].std():.3f}",
        'Precision': f"{cv_metrics['test_precision'].mean():.3f} ± {cv_metrics['test_precision'].std():.3f}",
        'Recall': f"{cv_metrics['test_recall'].mean():.3f} ± {cv_metrics['test_recall'].std():.3f}",
        'F1-Score': f"{cv_metrics['test_f1'].mean():.3f} ± {cv_metrics['test_f1'].std():.3f}",
        'Best Parameters': str(best_params_dict[name]) if name in best_params_dict else "BIC Optimized"
    })

cv_summary_df = pd.DataFrame(final_cv_results)

IPython.display.display(cv_summary_df)
cv_summary_df.to_csv('五折模型结果及参数.csv', encoding='utf-8-sig')

def plot_cv_roc_with_shadow(model_name, clf, X_data, y_data, prep,save_path=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(8, 7))
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
        
        fold_pipe = Pipeline([('prep', prep), ('clf', clf)])
        fold_pipe.fit(X_train_fold, y_train_fold)
        y_prob = fold_pipe.predict_proba(X_val_fold)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {aucs[-1]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('假阳率（FPR）')
    plt.ylabel('真阳率（TPR）')
    plt.title(f'{model_name} 模型 - 内部测试 ROC 曲线', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    plt.show()

logit_cont_cv = [f for f in logit_features if f in target_continuous_vars]
logit_prep = ColumnTransformer(
    transformers=[('num', StandardScaler(), logit_cont_cv)],
    remainder='passthrough'
)
plot_cv_roc_with_shadow('Logistic Regression', models['LogisticRegression'], 
                        X[logit_features], y, logit_prep, 
                        save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/Logistic内部测试ROC曲线.png')

plot_cv_roc_with_shadow('XGBoost', models['XGBoost'], 
                        X[final_selected_features], y, preprocessor,
                       save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/XGBoost内部测试ROC曲线.png')
plot_cv_roc_with_shadow('RandomForest', models['RandomForest'], 
                        X[final_selected_features], y, preprocessor,
                       save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/RandomForest内部测试ROC曲线.png')
plot_cv_roc_with_shadow('LightGBM', models['LightGBM'], 
                        X[final_selected_features], y, preprocessor,
                       save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/LightGBM内部测试ROC曲线.png')



plt.figure(figsize=(10, 8))
mean_fpr = np.linspace(0, 1, 100)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for idx, (name, clf) in enumerate(models.items()):
    tprs = []
    aucs = []
    
    if name == 'LogisticRegression':
        X_cv = X[logit_features]
        logit_cont_cv = [f for f in logit_features if f in target_continuous_vars]
        prep_cv = ColumnTransformer(
            transformers=[('num', StandardScaler(), logit_cont_cv)],
            remainder='passthrough'
        )
    else:
        X_cv = X[final_selected_features]
        prep_cv = preprocessor
    
    for train_idx, val_idx in skf_final.split(X_cv, y):
        X_train_f, X_val_f = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]
        
        fold_pipe = Pipeline([('prep', prep_cv), ('clf', clf)])
        fold_pipe.fit(X_train_f, y_train_f)
        y_prob = fold_pipe.predict_proba(X_val_f)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_val_f, y_prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)

    plt.plot(mean_fpr, mean_tpr, color=colors[idx],
             label=f'{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
             lw=2.5, alpha=.9)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[idx], alpha=.1)

# 装饰图片
plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='grey', label='', alpha=.6)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('假阳率（FPR）', fontsize=12)
plt.ylabel('真阳率（TPR）', fontsize=12)
plt.title('内部测试集：各模型 ROC 曲线对比', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('全模型ROC曲线.png', dpi=300, bbox_inches='tight')
plt.show()



# 外部验证：将外部数据集随机等分成五份，评估模型性能
#外部数据
file_path = 'MC_MED_Yuhou_EHR_Prognosis_Tw_6h.csv'
df_external = pd.read_csv(file_path)

label_col = 'LABEL'   
if label_col not in df_external.columns:
    raise ValueError(f"外部数据中未找到标签列 '{label_col}'，请检查列名。")

np.random.seed(42)
n_ext = len(df_external)
shuffled_idx = np.random.permutation(n_ext)
fold_sizes = [n_ext // 5] * 5
remainder = n_ext % 5
for i in range(remainder):
    fold_sizes[i] += 1

external_subsets = []
start = 0
for i, size in enumerate(fold_sizes):
    idx = shuffled_idx[start:start+size]
    external_subsets.append(df_external.iloc[idx].reset_index(drop=True))
    start += size
    print(f"外部子集 {i+1} 大小: {len(external_subsets[-1])}")

final_trained_pipelines = {}

for name, clf in models.items():
    if name == 'LogisticRegression':
        X_train_full = X[logit_features].copy()
        logit_cont_full = [f for f in logit_features if f in target_continuous_vars]
        prep_full = ColumnTransformer(
            transformers=[('num', StandardScaler(), logit_cont_full)],
            remainder='passthrough'
        )
    else:
        X_train_full = X[final_selected_features].copy()
        prep_full = preprocessor   
    
    full_pipe = Pipeline([('prep', prep_full), ('clf', clf)])
    full_pipe.fit(X_train_full, y)
    final_trained_pipelines[name] = full_pipe
    print(f"✓ {name} 模型训练完成")


external_cv_results = []

for name, pipe in final_trained_pipelines.items():
    aucs, auprcs, accs, precs, recs, f1s, briers = [], [], [], [], [], [], []
    
    for fold_df in external_subsets:
        if name == 'LogisticRegression':
            X_ext_fold = fold_df[logit_features]
        else:
            X_ext_fold = fold_df[final_selected_features]
        y_ext_fold = fold_df[label_col]
        
        y_prob = pipe.predict_proba(X_ext_fold)[:, 1]
        y_pred = pipe.predict(X_ext_fold)
        
        aucs.append(roc_auc_score(y_ext_fold, y_prob))
        auprcs.append(average_precision_score(y_ext_fold, y_prob))
        accs.append(accuracy_score(y_ext_fold, y_pred))
        precs.append(precision_score(y_ext_fold, y_pred, zero_division=0))
        recs.append(recall_score(y_ext_fold, y_pred, zero_division=0))
        f1s.append(f1_score(y_ext_fold, y_pred, zero_division=0))
        briers.append(brier_score_loss(y_ext_fold, y_prob))
    
    external_cv_results.append({
        'Model': name,
        'AUC': f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
        'AUPRC': f"{np.mean(auprcs):.3f} ± {np.std(auprcs):.3f}",
        'Brier': f"{np.mean(briers):.3f} ± {np.std(briers):.3f}",
        'Accuracy': f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        'Precision': f"{np.mean(precs):.3f} ± {np.std(precs):.3f}",
        'Recall': f"{np.mean(recs):.3f} ± {np.std(recs):.3f}",
        'F1-Score': f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
    })

# 显示外部验证结果表格
external_df = pd.DataFrame(external_cv_results)
display(external_df)
external_df.to_csv('外部验证_五份子集结果.csv',
                   encoding='utf-8-sig', index=False)

def plot_external_roc_individual(model_name, pipe, subsets, feature_cols, label_col, save_path=None):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    plt.figure(figsize=(8, 7))
    for i, fold_df in enumerate(subsets):
        X_fold = fold_df[feature_cols]
        y_fold = fold_df[label_col]
        y_prob = pipe.predict_proba(X_fold)[:, 1]
        fpr, tpr, _ = roc_curve(y_fold, y_prob)
        auc_val = auc(fpr, tpr)
        aucs.append(auc_val)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'子集 {i+1} (AUC = {auc_val:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='', alpha=.8)

    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('假阳率 (FPR)', fontsize=12)
    plt.ylabel('真阳率 (TPR)', fontsize=12)
    plt.title(f'{model_name} 模型 - 外部验证 ROC 曲线', fontsize=12)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ {model_name} ROC 曲线已保存至: {save_path}")
    plt.show()

def plot_external_roc_combined(trained_pipelines, subsets, feature_dict, label_col, save_path=None):
    mean_fpr = np.linspace(0, 1, 100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.figure(figsize=(10, 8))
    for idx, (name, pipe) in enumerate(trained_pipelines.items()):
        feature_cols = feature_dict[name]
        tprs = []
        aucs = []
        for fold_df in subsets:
            X_fold = fold_df[feature_cols]
            y_fold = fold_df[label_col]
            y_prob = pipe.predict_proba(X_fold)[:, 1]
            fpr, tpr, _ = roc_curve(y_fold, y_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=colors[idx], lw=2.5,
                 label=f'{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',alpha=.9)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[idx], alpha=.1)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5,color='grey', label='', alpha=0.6)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('假阳率 (FPR)', fontsize=12)
    plt.ylabel('真阳率 (TPR)', fontsize=12)
    plt.title('外部验证：各模型 ROC 曲线对比', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比 ROC 曲线已保存至: {save_path}")
    plt.show()

feature_dict = {
    'LogisticRegression': logit_features,
    'RandomForest': final_selected_features,
    'XGBoost': final_selected_features,
    'LightGBM': final_selected_features
}

# 绘制单个模型ROC
plot_external_roc_individual('LogisticRegression', final_trained_pipelines['LogisticRegression'],
                             external_subsets, logit_features, label_col,
                             save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/外部验证_LogisticROC.png')
plot_external_roc_individual('RandomForest', final_trained_pipelines['RandomForest'],
                             external_subsets, final_selected_features, label_col,
                             save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/外部验证_RandomForestROC.png')
plot_external_roc_individual('XGBoost', final_trained_pipelines['XGBoost'],
                             external_subsets, final_selected_features, label_col,
                             save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/外部验证_XGBoostROC.png')
plot_external_roc_individual('LightGBM', final_trained_pipelines['LightGBM'],
                             external_subsets, final_selected_features, label_col,
                             save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/外部验证_LightGBMROC.png')

# 绘制所有模型对比图
plot_external_roc_combined(final_trained_pipelines, external_subsets, feature_dict, label_col,
                           save_path='C:/Users/17207/Desktop/统计建模数据/EHR数据/图表/6h2/外部验证_全模型ROC对比.png')





#校准曲线

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


save_path = "校准曲线.png"
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

y_external = df_external[label_col]

plt.figure(figsize=(9, 7))
plt.plot([0, 1], [0, 1], "k:", label="精准线")

colors = plt.cm.tab10.colors

for idx, (name, pipe) in enumerate(final_trained_pipelines.items()):
    if name == 'LogisticRegression':
        X_ext = df_external[logit_features]
    else:
        X_ext = df_external[final_selected_features]
    prob_pos = pipe.predict_proba(X_ext)[:, 1]
    
    n_bins = 10
    fraction_pos, mean_pred = calibration_curve(
        y_external, prob_pos, n_bins=n_bins, strategy='uniform'  
    )
    
    brier = brier_score_loss(y_external, prob_pos)
    plt.plot(mean_pred, fraction_pos, 
             marker='o', markersize=6, linewidth=2,
             color=colors[idx % len(colors)], 
             label=f"{name} (Brier={brier:.3f})")

plt.ylabel("实际发生概率", fontsize=11)
plt.xlabel("预测概率", fontsize=11)
plt.title("校准曲线图", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()    


#SHAP
import shap
rf_pipe = final_trained_pipelines['RandomForest']
preprocessor = rf_pipe.named_steps['prep']
rf_model = rf_pipe.named_steps['clf']

try:
    feature_names_out = preprocessor.get_feature_names_out()
except:
    feature_names_out = final_selected_features


X_input = df_external[final_selected_features]

X_transformed = preprocessor.transform(X_input)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_transformed, check_additivity=False)
if isinstance(shap_values, list):
    final_shap_values = shap_values[1]
else:
    # 部分版本直接返回 3D 数组 [samples, features, classes]
    if len(shap_values.shape) == 3:
        final_shap_values = shap_values[:, :, 1]
    else:
        final_shap_values = shap_values

name_mapping = {
    'AGE': '年龄',
    'Glucose_mean': '血糖均值',
    'Urea_Nitrogen_last': '尿素氮末次值',
    'WBC_max': '白细胞最大值',
    'DBP_min': '舒张压最小值',
    'DBP_range': '舒张压极差',
    'HR_range': '心率极差',
    'RR_mean': '呼吸频率均值',
    'RR_min': '呼吸频率最小值',
    'RR_max': '呼吸频率最大值',
    'SBP_max': '收缩压最大值',
    'PP_mean': '脉压均值',
    'Shock_Index': '休克指数',
    'Shock_Index': '休克指数',
    'SPO2_mean':'外周血氧饱和度均值',
    'SPO2_min':'外周血氧饱和度最小值',
    'SPO2_max':'外周血氧饱和度最大值',
}

chinese_names = [name_mapping.get(name.split('__')[-1], name) for name in feature_names_out]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 8))
shap.summary_plot(
    final_shap_values, 
    X_transformed, 
    feature_names=chinese_names,
    plot_type="dot", 
    show=False
)

plt.title('随机森林外部验证：特征贡献度分析 (SHAP)', fontsize=15, pad=20)
plt.xlabel('SHAP 值 ', fontsize=12)
plt.savefig('随机森林外部SHAP.png', dpi=300, bbox_inches='tight')
plt.show()


X_input = X[final_selected_features]
X_transformed = preprocessor.transform(X_input)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()
plt.figure(figsize=(10, 8))
shap.summary_plot(
    final_shap_values, 
    X_transformed, 
    feature_names=chinese_names,
    plot_type="dot", 
    show=False
)
plt.title('随机森林内部验证：特征贡献度分析 (SHAP)', fontsize=15, pad=20)
plt.xlabel('SHAP 值 ', fontsize=12)
plt.savefig('随机森林内部SHAP.png', dpi=300, bbox_inches='tight')
plt.show()



