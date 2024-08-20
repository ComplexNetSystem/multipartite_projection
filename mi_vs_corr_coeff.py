import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import stats

# Rick wanted to compare MI with PCC. On July 15, 2024, by Jie
# Swallow plot.


df_discrete_yfs = pd.read_csv("yfs_discrete_data_only_sym_0712_filter_qcut_sturges_rand_imp_0.csv", index_col=0)

var_names = df_discrete_yfs.columns.values
dep_var = var_names[0:21].tolist()


last_meta_idx = np.where(var_names == 'bohbut07')[0][0]
last_risk_idx = np.where(var_names == 'met07')[0][0]  # phe includes sym and risks
first_cvd_idx = np.where(var_names == 'dkv07')[0][0]

var_group_meta = var_names[last_risk_idx + 1:last_meta_idx + 1].tolist()
var_group_lipid = var_names[last_meta_idx + 1:].tolist()
var_group_phen = var_names[:last_risk_idx + 1].tolist()   # including all CVD, depression and risk factors.
cvd_var = var_group_phen[first_cvd_idx:last_risk_idx]
exposures = var_group_phen[len(dep_var):first_cvd_idx] + [var_names[last_risk_idx]]
diseases = dep_var + cvd_var

pheno_sym_df = df_discrete_yfs.loc[:, diseases]
meta_df = df_discrete_yfs.loc[:, var_group_meta]

top10_meta = {"XLHDLPL07", "SLDLTG07", "Val07", "Crea07", "ApoBApoA107",
              "SVLDLPL07", "XSVLDLTG07", "LHDLFC07", "LLDLFCPCT07", "XXLVLDLTG07"}

top9_meta = {"XLHDLPL07", "SLDLTG07", "Val07", "ApoBApoA107",
              "SVLDLPL07", "XSVLDLTG07", "LHDLFC07", "LLDLFCPCT07", "XXLVLDLTG07"}


def mi_btn_groups(group1_df: pd.DataFrame, group2_df: pd.DataFrame, norm: bool):
    group1_vars = group1_df.columns.to_list()
    group2_vars = group2_df.columns.to_list()
    mi_values = np.array([])
    for var_in_1 in group1_vars:
        var1_array = group1_df[var_in_1].to_numpy()
        for var_in_2 in group2_vars:
            var2_array = group2_df[var_in_2].to_numpy()
            if norm:
                mi_val = metrics.normalized_mutual_info_score(var1_array, var2_array, average_method='min')
            else:
                mi_val = metrics.mutual_info_score(var1_array, var2_array)

            mi_values = np.append(mi_values, mi_val)

    return mi_values


def corr_btn_groups(group1_df: pd.DataFrame, group2_df: pd.DataFrame, method: str):
    group1_vars = group1_df.columns.to_list()
    group2_vars = group2_df.columns.to_list()
    corr_values = np.array([])
    for var_in_1 in group1_vars:
        var1_array = group1_df[var_in_1].to_numpy()
        for var_in_2 in group2_vars:
            var2_array = group2_df[var_in_2].to_numpy()
            if method == "Pearson":
                corr_val = stats.pearsonr(var1_array, var2_array)[0]
            elif method == "Spearman":
                corr_val = stats.spearmanr(var1_array, var2_array)[0]
            else:
                corr_val = None
                print("Method not included.")

            corr_values = np.append(corr_values, corr_val)

    return corr_values


def scatter_dot_colors(group1_df: pd.DataFrame, group2_df: pd.DataFrame, top_set: set):
    group1_vars = group1_df.columns.to_list()
    group2_vars = group2_df.columns.to_list()
    ii = 0
    color_dots = []
    color_highlight_index = []
    for var_in_1 in group1_vars:
        for var_in_2 in group2_vars:
            if {var_in_1, var_in_2}.intersection(top_set):
                color_dots.append("red")
                color_highlight_index.append(ii)
            else:
                color_dots.append("gray")
            ii = ii + 1
    return color_dots, color_highlight_index


mi_btn_sym_meta = mi_btn_groups(pheno_sym_df, meta_df, norm=False)
corr_btn_sym_meta = corr_btn_groups(pheno_sym_df, meta_df, method="Pearson")
colors_dots, highlight_idx = scatter_dot_colors(pheno_sym_df, meta_df, top9_meta)

# Scatter plot
# colors_dots = ['orange' if abs(corr) <= 0.1 else 'tab:blue' for corr in corr_btn_sym_meta]

mi_highlight = mi_btn_sym_meta[highlight_idx]
corr_highlight = corr_btn_sym_meta[highlight_idx]

plt.scatter(mi_btn_sym_meta, corr_btn_sym_meta, c=colors_dots, alpha=0.4)
plt.scatter(mi_highlight, corr_highlight, c='red', alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("MI", fontsize=20)
plt.ylabel("PCC", fontsize=20)

plt.tight_layout()
plt.savefig("PCC_vs_MI_color.pdf")
