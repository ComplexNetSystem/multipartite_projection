import pandas as pd
import numpy as np
import projection_methods as pm
import data_methods as dm
import matplotlib.pyplot as plt

# This is main projection codes for explaining and spurious score. April 22, 2024 by Jie.
# For new imputation with random samples.

alpha = 0.01
CASE = 'only_sym'
ABS_NORM = 'abs'

if CASE == 'only_sym':
    df_discrete_yfs = pd.read_csv("yfs_discrete_data_only_sym_0712_filter_qcut_sturges_rand_imp_0.csv", index_col=0)
    mi_norm = np.array(pd.read_csv("mi_norm_yfs_0712_only_sym_filter_qcut_sturges_rand_imp_0.csv", index_col=0))
    mi_abs = np.array(pd.read_csv("mi_abs_yfs_0712_only_sym_filter_qcut_sturges_rand_imp_0.csv", index_col=0))
    p_value_abs_mi = np.array(
        pd.read_csv("p_value_mi_yfs_chi2_0712_only_sym_filter_qcut_sturges_rand_imp_0.csv", index_col=0))
    var_names = df_discrete_yfs.columns.values
    dep_var = var_names[0:21].tolist()
elif CASE == 'only_summ':
    df_discrete_yfs = pd.read_csv("yfs_discrete_data_only_summ_0328_filter_qcut_sturges_rand_imp_10.csv", index_col=0)
    mi_norm = np.array(pd.read_csv("mi_norm_yfs_0328_only_summ_filter_qcut_sturges_rand_imp_10.csv", index_col=0))
    mi_abs = np.array(pd.read_csv("mi_abs_yfs_0328_only_summ_filter_qcut_sturges_rand_imp_10.csv", index_col=0))
    p_value_abs_mi = np.array(
        pd.read_csv("p_value_mi_yfs_chi2_0328_only_summ_filter_qcut_sturges_rand_imp_10.csv", index_col=0))
    var_names = df_discrete_yfs.columns.values
    dep_var = var_names[0:1].tolist()
else:
    print("CASE code error!")
    df_discrete_yfs = pd.DataFrame()
    mi_norm = []
    mi_abs = []
    p_value_abs_mi = []
    var_names = []
    dep_var = []

last_meta_idx = np.where(var_names == 'bohbut07')[0][0]
last_risk_idx = np.where(var_names == 'met07')[0][0]  # phe includes sym and risks
first_cvd_idx = np.where(var_names == 'dkv07')[0][0]

var_group_meta = var_names[last_risk_idx + 1:last_meta_idx + 1].tolist()
var_group_lipid = var_names[last_meta_idx + 1:].tolist()
var_group_phen = var_names[:last_risk_idx + 1].tolist()   # including all CVD, depression and risk factors.
cvd_var = var_group_phen[first_cvd_idx:last_risk_idx]
exposures = var_group_phen[len(dep_var):first_cvd_idx] + [var_names[last_risk_idx]]
diseases = dep_var + cvd_var

# mi_abs_p_val = p_value_abs_mi + p_value_abs_mi.T - np.diag(np.diag(p_value_abs_mi))
mi_abs_p_val = p_value_abs_mi.copy()

df_discrete_phen = df_discrete_yfs.loc[:, var_group_phen]
mi_norm_disease, mi_abs_disease = dm.cal_mi_skl_cluster(df_discrete_phen)
# np.fill_diagonal(mi_norm_disease, 0)
# np.fill_diagonal(mi_abs_disease, 0)
mi_abs_p_val_disease = dm.mi_p_val_chi2(df_discrete_phen)
# mi_abs_p_val_disease = mi_abs_p_val_disease + mi_abs_p_val_disease.T - np.diag(np.diag(mi_abs_p_val_disease))

# Bipartite Network.
level_bi_meta_list_dict = {'level_0': var_group_phen, 'level_1': var_group_meta}
level_bi_lipid_list_dict = {'level_0': var_group_phen, 'level_1': var_group_lipid}

if ABS_NORM == 'norm':
    mi_corr_all = np.copy(mi_norm)
    mi_corr_disease = np.copy(mi_norm_disease)
elif ABS_NORM == 'abs':
    mi_corr_all = np.copy(mi_abs)
    mi_corr_disease = np.copy(mi_abs_disease)
else:
    print("ABS_NORM code error!")
    mi_corr_all = np.zeros([len(var_names), len(var_names)])
    mi_corr_disease = np.zeros([len(var_group_phen), len(var_group_phen)])


alpha_list = np.linspace(0.01, 0.2, 20)

explain_score_p1, spurious_score_p1 = pm.score_alpha_func(mi_corr_disease, mi_abs_p_val_disease,
                                                          mi_corr_all, mi_abs_p_val,
                                                          alpha_list, 1, exposures, var_group_phen,
                                                          var_group_meta, var_group_lipid, var_names)
explain_score_p2, spurious_score_p2 = pm.score_alpha_func(mi_corr_disease, mi_abs_p_val_disease,
                                                          mi_corr_all, mi_abs_p_val,
                                                          alpha_list, 2, exposures, var_group_phen,
                                                          var_group_meta, var_group_lipid, var_names)
explain_score_p3, spurious_score_p3 = pm.score_alpha_func(mi_corr_disease, mi_abs_p_val_disease,
                                                          mi_corr_all, mi_abs_p_val,
                                                          alpha_list, 3, exposures, var_group_phen,
                                                          var_group_meta, var_group_lipid, var_names)

alpha_list_1 = np.linspace(0.01, 0.01, 1)
explain_score_p4, spurious_score_p4 = pm.score_alpha_func(mi_corr_disease, mi_abs_p_val_disease,
                                                          mi_corr_all, mi_abs_p_val,
                                                          alpha_list_1, 4, exposures, var_group_phen,
                                                          var_group_meta, var_group_lipid, var_names)

# alpha = 0.01 -> explain_score_p4, spurious_score_p4 = 0.0155, 0.8244769
# alpha = 0.05 -> explain_score_p4, spurious_score_p4 = 0.0155, 0.8244769
# alpha = 0.01 -> explain_score_p4, spurious_score_p4 = 0.013219, 0.848222   # Random 10. abs

plt.plot(alpha_list, explain_score_p1, label='Projection 1', color='#1f77b4')
plt.plot(alpha_list, explain_score_p2, label='Projection 2', color='red')
plt.plot(alpha_list, explain_score_p3, label='Projection 3', color='green')
plt.scatter(alpha_list_1, explain_score_p4, label='Projection 4', color='orange')
plt.xlabel(r'$ \alpha $', fontsize=18)
plt.ylabel('Explaining score', fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.xlim([-0.001, 0.202])
plt.ylim([-0.01, 0.51])
plt.legend()
plt.tight_layout()
plt.savefig("explaining_score_projection1234_0712_abs_p001_filter_qcut_sturges_rand_imp_0.pdf")

plt.plot(alpha_list, spurious_score_p1, label='Projection 1', color='#1f77b4')
plt.plot(alpha_list, spurious_score_p2, label='Projection 2', color='red')
plt.plot(alpha_list, spurious_score_p3, label='Projection 3', color='green')
plt.scatter(alpha_list_1, spurious_score_p4, label='Projection 4', color='orange')
plt.xlabel(r'$ \alpha $', fontsize=18)
plt.ylabel('Spurious score', fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.xlim([-0.001, 0.202])
plt.ylim([0.695, 1.005])
plt.yticks(np.arange(0.7, 1.005, 0.1))
plt.legend()
plt.tight_layout()
plt.savefig("spurious_score_projection1234_0712_abs_p001_filter_qcut_sturges_rand_imp_0.pdf")


# In a subplot.
n_col_fig = 2
fig, ax = plt.subplots(1, n_col_fig, figsize=(8, 4))

ax[0].plot(alpha_list, explain_score_p1, label='Projection 1', color='#1f77b4')
ax[0].plot(alpha_list, explain_score_p2, label='Projection 2', color='red')
ax[0].plot(alpha_list, explain_score_p3, label='Projection 3', color='green')
ax[0].scatter(alpha_list_1, explain_score_p4, label='Projection 4', color='orange')
# plt.ylim([-0.02, 0.42])
ax[0].tick_params(axis='both', which='major', labelsize=14)
ax[0].set_xlabel(r'$ \alpha $', fontsize=18)
ax[0].set_ylabel('Explaining score', fontsize=18)
ax[0].set_xlim([-0.001, 0.201])
ax[0].set_ylim([-0.01, 0.51])
ax[0].legend()

ax[1].plot(alpha_list, spurious_score_p1, label='Projection 1', color='#1f77b4')
ax[1].plot(alpha_list, spurious_score_p2, label='Projection 2', color='red')
ax[1].plot(alpha_list, spurious_score_p3, label='Projection 3', color='green')
ax[1].scatter(alpha_list_1, spurious_score_p4, label='Projection 4', color='orange')
ax[1].tick_params(axis='both', which='major', labelsize=14)
ax[1].set_xlabel(r'$ \alpha $', fontsize=18)
ax[1].set_ylabel('Spurious score', fontsize=18)
ax[1].set_xlim([-0.01, 0.21])
ax[1].set_yticks(np.arange(0.8, 0.97, 0.05))
ax[1].set_ylim([0.79, 0.96])
# ax[1].legend()

plt.tight_layout()
plt.savefig("explaining_spurious_score_projection1234_0823_filter_qcut.pdf")
