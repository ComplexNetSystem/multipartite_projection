import pandas as pd
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import networkx as nx
import seaborn as sns
import data_methods as dm
import graph_methods as gm
import matplotlib.pyplot as plt
import os

# This is main projection codes for new imputation. Mar. 28, 2024 by Jie
# Replace each missing value with a random sample from known values.
# Try a few times and make an average. April 8 by Jie

alpha = 0.01
CASE = 'only_sym'
ABS_NORM = 'abs'
SEX = "female"

if SEX == "female":
    file_suffix = "_female.csv"
elif SEX == "male":
    file_suffix = "_male.csv"
else:
    file_suffix = ".csv"

script_path = os.getcwd()
os.chdir(script_path + '/data_random_imp_0712_sex')

data_name = "yfs_discrete_data_" + CASE + "_0712_filter_qcut_sturges_rand_imp_"
mi_norm_name = "mi_norm_yfs_0712_" + CASE + "_filter_qcut_sturges_rand_imp_"
mi_abs_name = "mi_abs_yfs_0712_" + CASE + "_filter_qcut_sturges_rand_imp_"
p_val_name = "p_value_mi_yfs_chi2_0712_" + CASE + "_filter_qcut_sturges_rand_imp_"

# Define the variables and assign values from the first random imputation.
df_discrete_yfs = pd.read_csv(data_name + str(0) + file_suffix, index_col=0)
var_names = df_discrete_yfs.columns.values
dep_var = var_names[0:21].tolist()
last_meta_idx = np.where(var_names == 'bohbut07')[0][0]
last_risk_idx = np.where(var_names == 'met07')[0][0]  # phe includes sym and risks
first_cvd_idx = np.where(var_names == 'dkv07')[0][0]
var_group_meta = var_names[last_risk_idx + 1:last_meta_idx + 1].tolist()
var_group_lipid = var_names[last_meta_idx + 1:].tolist()
var_group_phen = var_names[:last_risk_idx + 1].tolist()  # including all CVD, depression and risk factors.
cvd_var = var_group_phen[first_cvd_idx:last_risk_idx]
exposures = var_group_phen[len(dep_var):first_cvd_idx] + [var_names[last_risk_idx]]
diseases = dep_var + cvd_var

# Initialize lists to saving the resulting data from N loops.
mi_norm_loops = []
mi_abs_loops = []
mi_p_value_loops = []
mi_norm_disease_loops = []
mi_abs_disease_loops = []
mi_p_value_disease_loops = []

cvd_dep_total_contribute_meta_2_loops = []
often_cvd_dep_contribute_meta_2_loops = []
joint_score_meta_loops = []
cvd_dep_total_contribute_lipid_2_loops = []
often_cvd_dep_contribute_lipid_2_loops = []
joint_score_lipid_loops = []

sn_contribute_meta_2_1_loops = []
sn_contribute_meta_2_2_loops = []
sn_contribute_lipid_2_1_loops = []
sn_contribute_lipid_2_2_loops = []

project_adj_meta_2_loops = []
project_adj_lipid_2_loops = []

N_imp_times = 20
loops = list(range(N_imp_times))
# loops = [0, 1, 2, 3, 4]
for ii in loops:
    if CASE == 'only_sym':
        df_discrete_yfs = pd.read_csv(data_name + str(ii) + file_suffix, index_col=0)
        mi_norm = np.array(pd.read_csv(mi_norm_name + str(ii) + file_suffix, index_col=0))
        mi_abs = np.array(pd.read_csv(mi_abs_name + str(ii) + file_suffix, index_col=0))
        p_value_abs_mi = np.array(pd.read_csv(p_val_name + str(ii) + file_suffix, index_col=0))
        var_names = df_discrete_yfs.columns.values
        dep_var = var_names[0:21].tolist()
    elif CASE == 'only_summ':
        df_discrete_yfs = pd.read_csv(data_name + str(ii) + file_suffix, index_col=0)
        mi_norm = np.array(pd.read_csv(mi_norm_name + str(ii) + file_suffix, index_col=0))
        mi_abs = np.array(pd.read_csv(mi_abs_name + str(ii) + file_suffix, index_col=0))
        p_value_abs_mi = np.array(pd.read_csv(p_val_name + str(ii) + file_suffix, index_col=0))
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

    mi_norm_loops.append(mi_norm)
    mi_abs_loops.append(mi_abs)
    mi_p_value_loops.append(p_value_abs_mi)

    last_meta_idx = np.where(var_names == 'bohbut07')[0][0]
    last_risk_idx = np.where(var_names == 'met07')[0][0]  # phe includes sym and risks
    first_cvd_idx = np.where(var_names == 'dkv07')[0][0]

    var_group_meta = var_names[last_risk_idx + 1:last_meta_idx + 1].tolist()
    var_group_lipid = var_names[last_meta_idx + 1:].tolist()
    var_group_phen = var_names[:last_risk_idx + 1].tolist()  # including all CVD, depression and risk factors.
    cvd_var = var_group_phen[first_cvd_idx:last_risk_idx]
    exposures = var_group_phen[len(dep_var):first_cvd_idx] + [var_names[last_risk_idx]]
    diseases = dep_var + cvd_var

    p_value_mi = p_value_abs_mi.copy()
    if ABS_NORM == 'norm':
        adj_mat = np.copy(mi_norm)
    elif ABS_NORM == 'abs':
        adj_mat = np.copy(mi_abs)
    else:
        print("ABS_NORM code error!")
        adj_mat = np.zeros([len(var_names), len(var_names)])
    adj_mat[p_value_mi > alpha] = 0
    np.fill_diagonal(adj_mat, 0)

    df_discrete_phen = df_discrete_yfs.loc[:, var_group_phen]
    mi_norm_disease, mi_abs_disease = dm.cal_mi_skl_cluster(df_discrete_phen)
    mi_abs_p_val_disease = dm.mi_p_val_chi2(df_discrete_phen)
    mi_norm_disease_loops.append(mi_norm_disease)
    mi_abs_disease_loops.append(mi_abs_disease)
    mi_p_value_disease_loops.append(mi_abs_p_val_disease)
    # mi_norm_disease[mi_abs_p_val_disease > alpha] = 0
    # mi_abs_disease[mi_abs_p_val_disease > alpha] = 0

    # Bipartite Network.
    level_bi_meta_list_dict = {'level_0': var_group_phen,
                               'level_1': var_group_meta}
    level_bi_lipid_list_dict = {'level_0': var_group_phen,
                                'level_1': var_group_lipid}
    bi_adj_df_meta, bi_G_meta = gm.bi_graph_create(adj_mat, var_group_phen, var_group_meta, var_names)
    bi_adj_df_lipid, bi_G_lipid = gm.bi_graph_create(adj_mat, var_group_phen, var_group_lipid, var_names)
    joint_score_meta = gm.joint_score_cal(bi_adj_df_meta, cvd_var, dep_var, var_group_meta)
    joint_score_lipid = gm.joint_score_cal(bi_adj_df_lipid, cvd_var, dep_var, var_group_lipid)
    joint_score_meta_loops.append(joint_score_meta)
    joint_score_lipid_loops.append(joint_score_lipid)

    # Metabolites
    project_adj_meta_2 = gm.adj_within_level_shared_neighbor_weighted(bi_G_meta, adj_mat, var_names,
                                                                      level_bi_meta_list_dict, 2)
    project_adj_meta_2_loops.append(project_adj_meta_2['level_0'])

    # Specific link: CVD-depression; risk-disease
    sn_contribute_meta_2_1 = gm.contribution_sn_weighted(bi_G_meta, adj_mat, var_names, 'bmi07', 'b18')
    sn_contribute_meta_2_2 = gm.contribution_sn_weighted(bi_G_meta, adj_mat, var_names, 'bmi07', 'idealCVH07')
    sn_contribute_meta_2_1_loops.append(sn_contribute_meta_2_1)
    sn_contribute_meta_2_2_loops.append(sn_contribute_meta_2_2)
    # sn_contribute_meta_2_3 = gm.contribution_sn_weighted(bi_G_meta, adj_mat, var_names, 'b21', 'idealCVH07')
    # sn_contribute_meta_2 = gm.contribution_sn_weighted(bi_G_meta, adj_mat, var_names, 'beckpisteet', 'idealCVH07')

    # All links between risk factors and diseases.
    total_risk_disease_contribute_meta_2 = gm.total_contribution_var_sn(bi_G_meta, adj_mat, var_names, exposures,
                                                                        diseases)
    often_risk_disease_contribute_meta_2 = gm.count_times_var_top_contribution(bi_G_meta, adj_mat, var_names, exposures,
                                                                               diseases, 10)

    # All links between depression and CVD variables.
    cvd_dep_total_contribute_meta_2 = gm.total_contribution_var_sn(bi_G_meta, adj_mat, var_names, dep_var, cvd_var)
    often_cvd_dep_contribute_meta_2 = gm.count_times_var_top_contribution(bi_G_meta, adj_mat, var_names, dep_var,
                                                                          cvd_var, 10)

    cvd_dep_total_contribute_meta_2_loops.append(cvd_dep_total_contribute_meta_2)
    often_cvd_dep_contribute_meta_2_loops.append(often_cvd_dep_contribute_meta_2)

    # Lipids
    project_adj_lipid_2 = gm.adj_within_level_shared_neighbor_weighted(bi_G_lipid, adj_mat, var_names,
                                                                       level_bi_lipid_list_dict, 2)
    project_adj_lipid_2_loops.append(project_adj_lipid_2['level_0'])

    # Specific link: CVD-depression; risk-disease
    sn_contribute_lipid_2_1 = gm.contribution_sn_weighted(bi_G_lipid, adj_mat, var_names, 'bmi07', 'b18')
    sn_contribute_lipid_2_2 = gm.contribution_sn_weighted(bi_G_lipid, adj_mat, var_names, 'bmi07', 'idealCVH07')
    sn_contribute_lipid_2_1_loops.append(sn_contribute_lipid_2_1)
    sn_contribute_lipid_2_2_loops.append(sn_contribute_lipid_2_2)
    # sn_contribute_lipid_2_3 = gm.contribution_sn_weighted(bi_G_lipid, adj_mat, var_names, 'b21', 'idealCVH07')
    # sn_contribute_lipid_2 = gm.contribution_sn_weighted(bi_G_lipid, adj_mat, var_names, 'beckpisteet', 'idealCVH07')

    # All links between risk factors and diseases.
    total_risk_disease_contribute_lipid_2 = gm.total_contribution_var_sn(bi_G_lipid, adj_mat, var_names, exposures,
                                                                         diseases)
    often_risk_disease_contribute_lipid_2 = gm.count_times_var_top_contribution(bi_G_lipid, adj_mat, var_names,
                                                                                exposures, diseases, 10)

    # All links between depression and CVD variables.
    cvd_dep_total_contribute_lipid_2 = gm.total_contribution_var_sn(bi_G_lipid, adj_mat, var_names, dep_var, cvd_var)
    often_cvd_dep_contribute_lipid_2 = gm.count_times_var_top_contribution(bi_G_lipid, adj_mat, var_names, dep_var,
                                                                           cvd_var, 10)

    cvd_dep_total_contribute_lipid_2_loops.append(cvd_dep_total_contribute_lipid_2)
    often_cvd_dep_contribute_lipid_2_loops.append(often_cvd_dep_contribute_lipid_2)

os.chdir(script_path)  # Go back to the main directory.

sn_contribute_meta_2_1_loops_df = pd.concat(sn_contribute_meta_2_1_loops).reset_index()
sn_contribute_meta_2_2_loops_df = pd.concat(sn_contribute_meta_2_2_loops).reset_index()
sn_contribute_lipid_2_1_loops_df = pd.concat(sn_contribute_lipid_2_1_loops).reset_index()
sn_contribute_lipid_2_2_loops_df = pd.concat(sn_contribute_lipid_2_2_loops).reset_index()

cvd_dep_total_contribute_meta_2_loops_df = pd.concat(cvd_dep_total_contribute_meta_2_loops).reset_index()
often_cvd_dep_contribute_meta_2_loops_df = pd.concat(often_cvd_dep_contribute_meta_2_loops).reset_index()
joint_score_meta_loops_df = pd.concat(joint_score_meta_loops).reset_index()
cvd_dep_total_contribute_lipid_2_loops_df = pd.concat(cvd_dep_total_contribute_lipid_2_loops).reset_index()
often_cvd_dep_contribute_lipid_2_loops_df = pd.concat(often_cvd_dep_contribute_lipid_2_loops).reset_index()
joint_score_lipid_loops_df = pd.concat(joint_score_lipid_loops).reset_index()

ttl_contribution_meta = cvd_dep_total_contribute_meta_2_loops_df['contribution'].to_list()
joint_score_meta = joint_score_meta_loops_df['score'].to_list()
ttl_contribution_lipid = cvd_dep_total_contribute_lipid_2_loops_df['contribution'].to_list()
joint_score_lipid = joint_score_lipid_loops_df['score'].to_list()

pcc_meta = []
spcc_meta = []
pcc_lipid = []
spcc_lipid = []
max_top = 51
step = 5
for top_n in range(10, max_top, step):
    pcc_meta.append(pearsonr(ttl_contribution_meta[: top_n], joint_score_meta[: top_n])[0])
    spcc_meta.append(spearmanr(ttl_contribution_meta[: top_n], joint_score_meta[: top_n])[0])
    pcc_lipid.append(pearsonr(ttl_contribution_lipid[: top_n], joint_score_lipid[: top_n])[0])
    spcc_lipid.append(spearmanr(ttl_contribution_lipid[: top_n], joint_score_lipid[: top_n])[0])

plt.plot(range(10, max_top, step), pcc_meta, 'o-', label='PCC (Metabolites)', color="deepskyblue")
plt.plot(range(10, max_top, step), spcc_meta, 'o-', label='SPCC (Metabolites)', color="tab:blue")
plt.plot(range(10, max_top, step), pcc_lipid, 'd-', label='PCC (Lipids)', color="lime")
plt.plot(range(10, max_top, step), spcc_lipid, 'd-', label='SPCC (Lipids)', color="forestgreen")
plt.legend(fontsize=12)
plt.xticks(range(0, max_top, 10), fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Top N", fontsize=20)
plt.ylabel("Correlation Coefficient", fontsize=20)
plt.tight_layout()
plt.savefig("topN_corr_mean_cont_jointness_p001_0712_filter_qcut_sturges_rand_imp_new_20_average.pdf")


bar_of_interest = 'contribution'  # contribution, joint, risk_dep, risk_cvd
val_col_name = 'contribution'
rank_of_interest_meta = cvd_dep_total_contribute_meta_2_loops_df
rank_of_interest_lipid = cvd_dep_total_contribute_lipid_2_loops_df
if bar_of_interest == 'joint':
    val_col_name = 'score'
    rank_of_interest_meta = joint_score_meta_loops_df
    rank_of_interest_lipid = joint_score_lipid_loops_df
elif bar_of_interest == 'risk_dep':
    val_col_name = 'contribution'
    rank_of_interest_meta = sn_contribute_meta_2_1_loops_df
    rank_of_interest_lipid = sn_contribute_lipid_2_1_loops_df
elif bar_of_interest == 'risk_cvd':
    val_col_name = 'contribution'
    rank_of_interest_meta = sn_contribute_meta_2_2_loops_df
    rank_of_interest_lipid = sn_contribute_lipid_2_2_loops_df
else:
    print("Keep the initial interested bar!")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
mean_values = rank_of_interest_meta.groupby('index')[val_col_name].mean().reset_index()
mean_values.rename(columns={val_col_name: 'mean'}, inplace=True)
top_mean_values = mean_values.nlargest(10, columns=['mean'])
std_values = rank_of_interest_meta.groupby('index')[val_col_name].sem().reset_index()
std_values.rename(columns={val_col_name: 'std'}, inplace=True)
# top_mean_std_values = pd.concat([top_mean_values, std_values], axis=1, join="inner")
top_mean_std_values = pd.merge(top_mean_values, std_values, on="index")
top_mean_std_values.fillna(0, inplace=True)
sns.barplot(ax=axes[0], x='mean', y='index', data=top_mean_std_values, xerr=top_mean_std_values['std'],
            color="dodgerblue", width=0.55)
# axes[0].set_xticks(np.arange(0, 0.12, 0.05))
axes[0].tick_params(axis='both', which='major', labelsize=16)
axes[0].set_xlabel("Mean total contribution", fontsize=20)
axes[0].set_ylabel("Metabolites", fontsize=20)

# Plot Female and Male together in one figure. Run for female and male,
# then save the "top_mean_std_values" respectively for female and male, and metabolites and lipids (see below).
mean_values.sort_values(by=['mean'], ascending=False, inplace=True)
top_mean_std_values = pd.merge(mean_values, std_values, on="index")
if SEX == "male":
    top_mean_std_values_meta_male = top_mean_std_values
elif SEX == "female":
    top_mean_std_values_meta_female = top_mean_std_values

mean_values = rank_of_interest_lipid.groupby('index')[val_col_name].mean().reset_index()
mean_values.rename(columns={val_col_name: 'mean'}, inplace=True)
top_mean_values = mean_values.nlargest(10, columns=['mean'])
std_values = rank_of_interest_lipid.groupby('index')[val_col_name].sem().reset_index()
std_values.rename(columns={val_col_name: 'std'}, inplace=True)
# top_mean_std_values = pd.concat([top_mean_values, std_values], axis=1, join="inner")
top_mean_std_values = pd.merge(top_mean_values, std_values, on="index")
top_mean_std_values.fillna(0, inplace=True)
sns.barplot(ax=axes[1], x='mean', y='index', data=top_mean_std_values, xerr=top_mean_std_values['std'],
            color="limegreen", width=0.55)
axes[1].tick_params(axis='both', which='major', labelsize=16)
# axes[1].set_xticks(np.arange(0, 0.16, 0.05))
axes[1].set_xlabel("Mean total contribution", fontsize=20)
axes[1].set_ylabel("Lipids", fontsize=20)
plt.tight_layout()
plt.savefig("top10_mean_ttl_cont_p001_0712_filter_qcut_sturges_rand_imp_new_20_average_male.pdf")


# Plot Female and Male together in one figure. Run for female and male,
# then save the "top_mean_std_values" respectively for female and male, and metabolites (see above) and lipids.
mean_values.sort_values(by=['mean'], ascending=False, inplace=True)
top_mean_std_values = pd.merge(mean_values, std_values, on="index")
if SEX == "male":
    top_mean_std_values_lipid_male = top_mean_std_values
elif SEX == "female":
    top_mean_std_values_lipid_female = top_mean_std_values


# Select top 10 mean values
top_N_mean_ttl = 10
top_male = top_mean_std_values_meta_male.nlargest(top_N_mean_ttl, 'mean')
top_female = top_mean_std_values_meta_female.nlargest(top_N_mean_ttl, 'mean')
top_male_in_female = top_male[['index']].merge(top_mean_std_values_meta_female, on='index', how='left')
top_female_in_male = top_female[['index']].merge(top_mean_std_values_meta_male, on='index', how='left')

top_gender = top_female.copy()
top_gender_the_other = top_female_in_male.copy()

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 8))
bar_width = 0.35
# Set positions for bars
female_positions = np.arange(top_N_mean_ttl)
male_positions = np.arange(top_N_mean_ttl)

# Plot bars for males and females
bars1 = ax.barh(female_positions - bar_width/2, top_gender['mean'], bar_width, xerr=top_gender['std'], label='Female', color='blue', capsize=5, align='center')  # dodgerblue; lightgreen
bars2 = ax.barh(male_positions + bar_width/2, top_gender_the_other['mean'], bar_width, xerr=top_gender_the_other['std'], label='Male', color='dodgerblue', capsize=5, align='center')     # blue; forestgreen

# Combine categories for y-tick labels
combined_categories = list(top_gender['index'])

# Set yticks and labels
# ax.set_xticks(np.arange(0, 0.41, 0.1))
ax.set_yticks(male_positions)
ax.set_yticklabels(combined_categories, fontsize=16)
# Add labels, title, and legend
ax.set_xlabel('Mean total contribution', fontsize=20)
ax.set_ylabel('Lipids', fontsize=20)
# ax.set_title('Top 10 Mean Values for Male and Female with Standard Deviations')
plt.xticks(fontsize=16)
plt.gca().invert_yaxis()
ax.legend(["Female", "Male"], fontsize=16)
# ax.legend(["Male", "Female"], fontsize=16)
plt.tight_layout()
plt.savefig("top10_meta_mean_ttl_cont_p001_0712_filter_qcut_sturges_20_average_female_index_swap_color.pdf")

# Old one by Jie
# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 8))
bar_width = 0.35
# Set positions for bars
female_positions = np.arange(len(top_female))
male_positions = np.arange(len(top_male))

# Plot bars for males and females
bars1 = ax.barh(female_positions - bar_width/2, top_male['mean'], bar_width, xerr=top_male['std'], label='Female', color='limegreen', capsize=5, align='center')  # dodgerblue
bars2 = ax.barh(male_positions + bar_width/2, top_female['mean'], bar_width, xerr=top_female['std'], label='Male', color='orange', capsize=5, align='center')     # darkorchid

# Combine categories for y-tick labels
combined_categories = list(top_female['index']) + list(top_male['index'])
combined_positions = np.concatenate([female_positions - bar_width/2, male_positions + bar_width/2])

# Set yticks and labels
# ax.set_xticks(np.arange(0, 0.41, 0.1))
ax.set_yticks(combined_positions)
ax.set_yticklabels(combined_categories, fontsize=16)
# Add labels, title, and legend
ax.set_xlabel('Mean total contribution', fontsize=20)
ax.set_ylabel('Lipids', fontsize=20)
# ax.set_title('Top 10 Mean Values for Male and Female with Standard Deviations')
plt.xticks(fontsize=16)
plt.gca().invert_yaxis()
ax.legend(["Female", "Male"], fontsize=16)
plt.tight_layout()
plt.savefig("top10_lipid_mean_ttl_cont_p001_0712_filter_qcut_sturges_rand_imp_new_20_average_female_male.pdf")


# Averaging MI and projected correlation and p-values as adjacency matrix.
mean_mi_abs = np.mean(np.array(mi_abs_loops), axis=0)
mean_mi_abs_p_val = np.mean(np.array(mi_p_value_loops), axis=0)

mean_mi_abs_disease = np.mean(np.array(mi_abs_disease_loops), axis=0)
mean_mi_abs_p_val_disease = np.mean(np.array(mi_p_value_disease_loops), axis=0)

mean_project_adj_meta_2 = np.mean(np.array(project_adj_meta_2_loops), axis=0)
mean_project_adj_lipid_2 = np.mean(np.array(project_adj_lipid_2_loops), axis=0)
mean_project_adj_meta_lipid_2 = mean_project_adj_meta_2 + mean_project_adj_lipid_2

# 1. Full tripartite network.

adj_mat_tri = mean_mi_abs.copy()
adj_mat_tri[mean_mi_abs_p_val > alpha] = 0

for ii in range(len(var_names)):
    for jj in range(len(var_names)):
        var_ii = var_names[ii]
        var_jj = var_names[jj]
        if (var_ii in var_group_phen and var_jj in var_group_phen) or \
                (var_ii in var_group_meta and var_jj in var_group_meta) or \
                (var_ii in var_group_lipid and var_jj in var_group_lipid) or \
                (var_ii in var_group_meta and var_jj in var_group_lipid) or \
                (var_ii in var_group_lipid and var_jj in var_group_meta):
            adj_mat_tri[ii, jj] = 0

# For full network, add this filtering, NO need, can be done on Gephi.
# top = 0.2
# threshold = gm.net_mat_threshold(adj_mat_tri, top, is_directed=False)
# adj_mat_tri[adj_mat_tri < threshold] = 0

df_adj_tri = pd.DataFrame(adj_mat_tri, index=var_names, columns=var_names)
G = nx.from_pandas_adjacency(df_adj_tri)
G.remove_nodes_from(list(nx.isolates(G)))
# G.remove_nodes_from(list(set(list(nx.isolates(G))) - set(var_group_phen)))
print("The number of nodes in the network is", len(G.nodes))
# Gephi data preparation
nx.write_gexf(G, "tripartite_YFS_symptoms_depress_MI_corr_p001_rand_imp_average_0712.gexf")

var_groups = [dep_var, cvd_var, var_group_meta, var_group_lipid, exposures]
node_table = pd.read_csv("nodes_tbl_tri.csv")
gm.add_node_attribute(node_table, var_groups, 'node_group')
node_table.to_csv("nodes_tbl_tri_group.csv", index=False)

"""
# Commented codes are for tripartite network viz. will use gephi for the visualization instead.
edges = list(G.edges)
weight = np.array([G[u][v]['weight'] for u, v in G.edges])
weight_dict = dict(zip(edges, weight))
node_fun_degree = np.array(list(dict(G.degree(weight="weight")).values()))
# No risk factors
# G.remove_nodes_from(exposures)
# G.remove_nodes_from(list(nx.isolates(G)))
# print("The number of nodes in the network is", len(G.nodes))

phe_nodes = []
meta_nodes = []
lipid_nodes = []
risk_nodes = []
color_nodes = []
for node in G.nodes:
    if node in var_group_phen:
        phe_nodes.append(node)
        if node in dep_var:
            color_nodes.append('red')
        elif node in cvd_var:
            color_nodes.append('gold')
        elif node in exposures:
            risk_nodes.append(node)
            color_nodes.append('orchid')
        else:
            print('Node label error!')
    elif node in var_group_meta:
        meta_nodes.append(node)
        color_nodes.append('tab:blue')
    elif node in var_group_lipid:
        lipid_nodes.append(node)
        color_nodes.append('tab:green')
    else:
        print('Node group error!')

G_tri = nx.complete_multipartite_graph(meta_nodes, phe_nodes, lipid_nodes)
pos_tri = nx.multipartite_layout(G_tri)
node_str_degree = np.array([x[1] for x in list(G.degree)])
plt.figure(figsize=[2, 4])
nx.draw_networkx_nodes(G, pos_tri, node_size=node_fun_degree/max(node_fun_degree)*1000, node_color=color_nodes)
nx.draw_networkx_edges(G, pos_tri, width=weight, alpha=0.2)
plt.axis('off')
plt.tight_layout()
plt.savefig("tri_net_only_summ_p001_0823_filter_qcut.pdf")
"""

# 2. Projected Network viz.

adj_mat_projected = mean_project_adj_lipid_2.copy()

# %%---- Disease network including CVD, depression and risk factors.
for ii in range(len(adj_mat_projected)):
    for jj in range(len(adj_mat_projected)):
        var_ii = var_group_phen[ii]
        var_jj = var_group_phen[jj]
        if (var_ii in cvd_var and var_jj in cvd_var) or \
                (var_ii in dep_var and var_jj in dep_var) or \
                (var_ii in exposures and var_jj in exposures) or \
                (var_ii in cvd_var and var_jj in dep_var) or \
                (var_ii in dep_var and var_jj in cvd_var):
            adj_mat_projected[ii, jj] = 0

# %% Disease network including CVD, depression.
for ii in range(len(adj_mat_projected)):
    for jj in range(len(adj_mat_projected)):
        var_ii = var_group_phen[ii]
        var_jj = var_group_phen[jj]
        if (var_ii in cvd_var and var_jj in cvd_var) or \
                (var_ii in dep_var and var_jj in dep_var) or \
                (var_ii in exposures and var_jj in exposures) or \
                (var_ii in exposures and var_jj in dep_var) or \
                (var_ii in dep_var and var_jj in exposures) or \
                (var_ii in exposures and var_jj in cvd_var) or \
                (var_ii in cvd_var and var_jj in exposures):
            adj_mat_projected[ii, jj] = 0


df_adj_projected = pd.DataFrame(adj_mat_projected, index=var_group_phen, columns=var_group_phen)
G = nx.from_pandas_adjacency(df_adj_projected)
# G.remove_nodes_from(list(nx.isolates(G)))
# G.remove_nodes_from(exposures)
print("The number of nodes in the network is", len(G.nodes))

node_str_degree = np.array([x[1] for x in list(G.degree)])
# node_fun_degree = adj_mat.sum(axis=0)
# node_fun_degree = node_fun_degree / max(node_fun_degree)
edges = list(G.edges)
weight = np.array([G[u][v]['weight'] for u, v in G.edges])
weight_dict = dict(zip(edges, weight))
node_fun_degree = np.array(list(dict(G.degree(weight="weight")).values()))

cvd_nodes = []
risk_nodes = []
dep_nodes = []
color_nodes = []
for node in G.nodes:
    if node in dep_var:
        dep_nodes.append(node)
        color_nodes.append('red')
    elif node in cvd_var:
        cvd_nodes.append(node)
        color_nodes.append('gold')
    elif node in exposures:
        risk_nodes.append(node)
        color_nodes.append('orchid')
    else:
        print("Node label error!")

plt.figure(figsize=(4.2, 8))
G_tri = nx.complete_multipartite_graph(dep_nodes, risk_nodes, cvd_nodes)
pos_tri = nx.multipartite_layout(G_tri, align='vertical')
# pos_bi = nx.bipartite_layout(G, dep_nodes, align='vertical')
nx.draw_networkx_nodes(G, pos=pos_tri, node_size=node_fun_degree / max(node_fun_degree) * 600, node_color=color_nodes)
nx.draw_networkx_edges(G, pos=pos_tri, width=weight / max(weight) * 5, edge_color='tab:green', alpha=0.4)  # tab: bl/gr
nx.draw_networkx_labels(G, pos=pos_tri, font_size=16, horizontalalignment='center')
plt.axis("off")
plt.tight_layout()
plt.savefig("proj_2_lipid_sym_risk_p001_0712_filter_qcut_rand_imp_new_20_average_female_1.pdf")

contribute_exposure = df_adj_projected.loc[exposures, diseases]
df_contribute_exposure_per = contribute_exposure.div(contribute_exposure.sum(axis=0))
df_contribute_exposure_per.iloc[:, :21].mean(axis=1)
df_contribute_exposure_per.iloc[:, 21:].mean(axis=1)


# projected corr VS. MI corr
mi_corr_disease = mean_mi_abs_disease.copy()
proj_corr_disease = mean_project_adj_meta_lipid_2.copy()

for ii in range(len(mi_corr_disease)):
    for jj in range(len(mi_corr_disease)):
        var_ii = var_group_phen[ii]
        var_jj = var_group_phen[jj]
        if (var_ii in cvd_var and var_jj in cvd_var) or \
                (var_ii in dep_var and var_jj in dep_var) or \
                (var_ii in exposures and var_jj in exposures) or \
                (var_ii in cvd_var and var_jj in dep_var) or \
                (var_ii in dep_var and var_jj in cvd_var):
            mi_corr_disease[ii, jj] = 0

for ii in range(len(proj_corr_disease)):
    for jj in range(len(proj_corr_disease)):
        var_ii = var_group_phen[ii]
        var_jj = var_group_phen[jj]
        if (var_ii in cvd_var and var_jj in cvd_var) or \
                (var_ii in dep_var and var_jj in dep_var) or \
                (var_ii in exposures and var_jj in exposures) or \
                (var_ii in cvd_var and var_jj in dep_var) or \
                (var_ii in dep_var and var_jj in cvd_var):
            proj_corr_disease[ii, jj] = 0

# Log-log: between MI and projected correlation.
mi_abs_val_arr = mi_corr_disease[np.triu_indices(len(mi_corr_disease), 1)]
proj_corr_val_arr = proj_corr_disease[np.triu_indices(len(proj_corr_disease), 1)]
x = mi_abs_val_arr / max(mi_abs_val_arr)
y = proj_corr_val_arr / max(proj_corr_val_arr)
x_nonzero = x[(x != 0) & (y != 0)]
y_nonzero = y[(x != 0) & (y != 0)]
x = np.log(x_nonzero)
y = np.log(y_nonzero)
data_for_reg = pd.DataFrame({"MI": x, "Proj": y})

# plt.figure(figsize=(6.5, 4.5))
ax = sns.regplot(data_for_reg, x="MI", y="Proj", line_kws={"color": "black", "alpha": 0.6})
slope_lm, intercept_lm, r_pearson, p_val_lm, sterr = scipy.stats.linregress(x, y)
r_squared = r2_score(x, y)
ax.text(-6.4, -0.6, f'$y = {str(round(intercept_lm, 2))} + {str(round(slope_lm, 2))}x$', fontsize=16)
ax.text(-6.4, -1.6, f'$p = {format(p_val_lm, ".2e")}$', fontsize=16)
ax.text(-6.4, -2.6, f'$r = {r_pearson:.2f}$', fontsize=16)
# ax.text(-6.4, -2.5, f'$R^2 = {r_squared:.2f}$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel('ln($MI$)', fontsize=20)
plt.ylabel('ln($Corr_{Projected}$)', fontsize=20)
plt.tight_layout()
plt.savefig("proj_2_ABS_proj_corr_vs_MI_loglog_only_sym_0712_filter_qcut_rand_imp_new_20_average_linear_reg.pdf")


plt.scatter(x, y, s=60, alpha=0.6)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color='k', linewidth=3, alpha=0.6)
# plt.axline((0, 0), slope=1, c='k')
plt.xlabel('log(corr)$_{MI}$', fontsize=20)
plt.ylabel('log(corr)$_{Projected}$', fontsize=20)
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig("proj_2_ABS_proj_corr_vs_MI_loglog_only_sym_0424_filter_qcut_rand_imp_new_20_average.pdf")

# Linear: between MI and projected correlation.
mi_abs_val_arr = mi_corr_disease[np.triu_indices(len(mi_corr_disease), 1)]
proj_corr_val_arr = proj_corr_disease[np.triu_indices(len(proj_corr_disease), 1)]
plt.scatter(mi_abs_val_arr / max(mi_abs_val_arr), proj_corr_val_arr / max(proj_corr_val_arr), s=60, alpha=0.6)
plt.xlabel('MI correlation', fontsize=20)
plt.ylabel('Projected correlation', fontsize=20)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig("proj_2_corr_vs_MI_tri_only_sym_0328_filter_qcut_sturges_impute.pdf")

# mi_corr_disease_df = pd.DataFrame(mi_corr_disease, index=var_group_phen, columns=var_group_phen)
# proj_corr_disease_df = pd.DataFrame(proj_corr_disease, index=var_group_phen, columns=var_group_phen)

plt.figure(figsize=(20, 11.5))
ax = sns.heatmap(mi_corr_disease, cmap='Spectral_r', xticklabels=var_group_phen, yticklabels=var_group_phen,
                 square=True)
ax.tick_params(labelsize=18)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig("norm_MI_phen_matrix_only_sym_0328_filtering_sturges_impute.pdf")

plt.figure(figsize=(20, 11.5))
ax = sns.heatmap(proj_corr_disease, cmap='Spectral_r', xticklabels=var_group_phen, yticklabels=var_group_phen,
                 square=True)
ax.tick_params(labelsize=18)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig("proj_phen_matrix_only_sym_0328_filtering_sturges_impute.pdf")

# Functional degree comparison
node_fun_degree_mi = mi_corr_disease.sum(axis=0)
node_fun_degree_proj = proj_corr_disease.sum(axis=0)
x = node_fun_degree_mi / max(node_fun_degree_mi)
y = node_fun_degree_proj / max(node_fun_degree_proj)

x_nonzero = x[(x != 0) & (y != 0)]
y_nonzero = y[(x != 0) & (y != 0)]
x = np.log(x_nonzero)
y = np.log(y_nonzero)

plt.scatter(x, y, s=60, alpha=0.6)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color='k', linewidth=3, alpha=0.6)
plt.xlabel('log(Weighted degree)$_{MI}$', fontsize=20)
plt.ylabel('log(Weighted degree)$_{Projected}$', fontsize=20)
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig("proj_2_ABS_proj_degree_vs_MI_degree_loglog_only_sym_0328_filter_qcut_sturges_impute.pdf")
