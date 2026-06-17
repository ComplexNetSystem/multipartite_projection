import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import data_methods as dm
import graph_methods as gm
import matplotlib.pyplot as plt
import itertools

# This is main projection codes for Pashu's DNA methylation project.
# The new version has excluded chrX and chrY.
# Sep 19, 2025 by Jie

alpha = 0.01
ABS_NORM = 'abs'
SEX = ""
suffix = '.csv'
if SEX == "Male":
    suffix = '_male.csv'
elif SEX == "Female":
    suffix = '_female.csv'
else:
    print("No gender specified!")

df_discrete_yfs = pd.read_csv("YFS_2018_2020_updated/yfs_discrete_18_20_0919_qcut_sturges" + suffix, index_col=0)
mi_norm = np.array(pd.read_csv("YFS_2018_2020_updated/mi_norm_yfs_18_20_0919_qcut_sturges" + suffix, index_col=0))
mi_abs = np.array(pd.read_csv("YFS_2018_2020_updated/mi_abs_yfs_18_20_0919_qcut_sturges" + suffix, index_col=0))
p_value_abs_mi = np.array(pd.read_csv("YFS_2018_2020_updated/mi_chi2_p_val_yfs_18_20_0919_qcut_sturges" + suffix, index_col=0))
var_names = df_discrete_yfs.columns.values

last_risk_phen_idx = np.where(var_names == 'Smoking')[0][0]
vars_risk_phen = var_names[:last_risk_phen_idx + 1].tolist()
vars_dna_meth = var_names[last_risk_phen_idx + 1:].tolist()

# Variable group TBD.
exposures = ['Age', 'Sex', 'BMI', 'CRP', 'MET', 'HDL_cholesterol', 'Total_cholesterol',
             'Systolic_BP', 'Triglycerides', 'Alcohol_consumption', 'Smoking']
if SEX == "Male" or SEX == "Female":
    exposures.remove("Sex")

# "MAFLD" or "NAFLD", in this version: "MAFLD"
diseases = ['cIMT', 'cPLAQUE', 'MAFLD', 'Creatinine', 'T2Diabetes', 'Rheumatic_arthritis', 'Asthma',
            'Clinical_depression', 'Anxiety', 'Osteoarthritis']

risk_disease_links = list(itertools.product(exposures, diseases))
disease_disease_links = list(itertools.combinations(diseases, 2))
risk_disease_disease_links = risk_disease_links + disease_disease_links

dep_vars = ['Clinical_depression', 'Anxiety']
cvd_vars = ['cIMT', 'cPLAQUE']

p_value_mi = p_value_abs_mi
if ABS_NORM == 'norm':
    corr_matrix = np.copy(mi_norm)
elif ABS_NORM == 'abs':
    corr_matrix = np.copy(mi_abs)
else:
    print("ABS_NORM code error!")
    corr_matrix = np.zeros([len(var_names), len(var_names)])

adj_mat = corr_matrix.copy()
adj_mat[p_value_mi > alpha] = 0
np.fill_diagonal(adj_mat, 0)

df_discrete_phen = df_discrete_yfs.loc[:, vars_risk_phen]
mi_norm_disease, mi_abs_disease = dm.cal_mi_skl_cluster(df_discrete_phen)
np.fill_diagonal(mi_norm_disease, 0)
np.fill_diagonal(mi_abs_disease, 0)
mi_abs_p_val_disease = dm.mi_p_val_chi2(df_discrete_phen)
# mi_abs_p_val_disease = mi_abs_p_val_disease + mi_abs_p_val_disease.T - np.diag(np.diag(mi_abs_p_val_disease))
mi_norm_disease[mi_abs_p_val_disease > alpha] = 0
mi_abs_disease[mi_abs_p_val_disease > alpha] = 0

# Bipartite Network.
level_bi_list_dict = {'level_0': vars_risk_phen,
                      'level_1': vars_dna_meth}

bi_adj_df_meth, bi_G_meth = gm.bi_graph_create(adj_mat, vars_risk_phen, vars_dna_meth, var_names)

# joint_score_meth = gm.joint_score_cal(bi_adj_df_meth, cvd_var, dep_var, vars_dna_meth)
# project_adj_meth = gm.adj_within_level_shared_neighbor_weighted(bi_G_meth, adj_mat, var_names, level_bi_list_dict, 2)

# Specific link: CVD-depression; risk-disease
cont_meth_links = pd.DataFrame(index=vars_dna_meth)
for proj_pair in risk_disease_disease_links:
    print(f'The pair is {proj_pair}')
    sn_contribute_specific_link = gm.contribution_sn_weighted(bi_G_meth, adj_mat, var_names, proj_pair[0], proj_pair[1])
    sn_contribute_specific_link_reind = sn_contribute_specific_link.reindex(cont_meth_links.index, fill_value=0)
    cont_meth_links[proj_pair] = sn_contribute_specific_link_reind.iloc[:, 0]
# cont_meth_links.to_excel("meth_contribution_each_link_" + SEX + "_p005_20251230.xlsx", index=True)


# All links between risk factors and diseases.
ttl_contribute_risk_meth = gm.total_contribution_var_sn(bi_G_meth, adj_mat, var_names, exposures, diseases)
often_contribute_risk_meth = gm.count_times_var_top_contribution(bi_G_meth, adj_mat, var_names, exposures, diseases, 10)
# ttl_contribute_risk_meth.to_excel("meth_contribution_risk_disease_" + SEX + "_p005_20251230.xlsx", index=True)

# # All links between depression and CVD variables.
ttl_contribute_comorbidity_meth = gm.total_contribution_var_sn(bi_G_meth, adj_mat, var_names, dep_vars, cvd_vars)
often_cont_comorbidity_meth = gm.count_times_var_top_contribution(bi_G_meth, adj_mat, var_names,
                                                                  dep_vars, cvd_vars, 10)
# ttl_contribute_comorbidity_meth.to_excel("meth_contribution_cvd_depression_" + SEX + "_p005_20251230.xlsx", index=True)

# All links between risk factors and diseases.
ttl_contribute_diseases_meth = gm.total_contribution_var_sn(bi_G_meth, adj_mat, var_names, diseases, diseases)
often_contribute_diseases_meth = gm.count_times_var_top_contribution(bi_G_meth, adj_mat, var_names, diseases, diseases, 10)
# ttl_contribute_diseases_meth.to_excel("meth_contribution_all_diseases_" + SEX + "_p005_20251230.xlsx", index=True)


# # Ranking visualization: bar plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

often_top = ttl_contribute_risk_meth.nlargest(10, columns=['contribution'])
ax = often_top.plot(kind='barh', fontsize=16, ax=axes[0])
ax.invert_yaxis()
# ax.set_xticks(np.arange(0, 0.51, 0.1))
# ax.set_title("Top 20 closeness centrality", fontsize=16)
ax.set_xlabel("Total contribution", fontsize=20)
ax.set_ylabel("DNA Methylation", fontsize=20)
ax.legend().remove()

often_top = often_contribute_risk_meth.nlargest(10, columns=['times'])
ax = often_top.plot(kind='barh', fontsize=16, ax=axes[1])
ax.invert_yaxis()
# ax.set_xticks(np.arange(0, 5, 1))
# ax.set_title("Top 20 closeness centrality", fontsize=16)
ax.set_xlabel("Frequency", fontsize=20)
ax.set_ylabel("DNA Methylation", fontsize=20)
ax.legend().remove()
plt.tight_layout()
# plt.savefig("top10_ttl_dna_contribute_1230_" + SEX + ".pdf")


# P-value calculation + null + permuting the meth variables' order. Dec. 30
n_permutation = 1000
adj_mat_df = pd.DataFrame(adj_mat, index=df_discrete_yfs.columns.values, columns=df_discrete_yfs.columns.values)
keep_n_meth_nodes = ttl_contribute_risk_meth.index.to_list()[:1000]
keep_nodes = vars_risk_phen + keep_n_meth_nodes
adj_mat_df = adj_mat_df.loc[keep_nodes, keep_nodes]
adj_mat_4_permute = adj_mat_df.to_numpy()

vars_dna_meth_arr = np.asarray(keep_n_meth_nodes)
ttl_contribute_risk_meth_n = ttl_contribute_risk_meth.copy()
ttl_contribute_comorbidity_meth_n = ttl_contribute_comorbidity_meth.copy()
ttl_contribute_diseases_meth_n = ttl_contribute_diseases_meth.copy()
for i in range(n_permutation):
    print(f"N permutation: {i}")
    permuted_dna_meth = np.random.permutation(vars_dna_meth_arr).tolist()
    var_names = np.array(vars_risk_phen + permuted_dna_meth)
    bi_adj_df_meth, bi_G_meth = gm.bi_graph_create(adj_mat_4_permute, vars_risk_phen, keep_n_meth_nodes, var_names)
    # All links between risk factors and diseases.
    ttl_contribute_risk_meth_i = gm.total_contribution_var_sn(bi_G_meth, adj_mat_4_permute, var_names, exposures, diseases)
    if not ttl_contribute_risk_meth_i.empty:
        ttl_contribute_risk_meth_n = pd.concat(
            [ttl_contribute_risk_meth_n, ttl_contribute_risk_meth_i], axis=1
        )
    # All links between depression and CVD variables.
    ttl_contribute_comorbidity_meth_i = gm.total_contribution_var_sn(bi_G_meth, adj_mat_4_permute, var_names, dep_vars, cvd_vars)
    if not ttl_contribute_comorbidity_meth_i.empty:
        ttl_contribute_comorbidity_meth_n = pd.concat(
            [ttl_contribute_comorbidity_meth_n, ttl_contribute_comorbidity_meth_i], axis=1
        )
    # All links between risk factors and diseases.
    ttl_contribute_diseases_meth_i = gm.total_contribution_var_sn(bi_G_meth, adj_mat_4_permute, var_names, diseases, diseases)
    if not ttl_contribute_diseases_meth_i.empty:
        ttl_contribute_diseases_meth_n = pd.concat(
            [ttl_contribute_diseases_meth_n, ttl_contribute_diseases_meth_i], axis=1
        )

p_val_top_n = 20
p_val_list = []
top_n_var_contribute = ttl_contribute_risk_meth_n.copy()
for ii in range(p_val_top_n):
    top_ii_permute = top_n_var_contribute.iloc[ii, :].values
    real_contribution = top_ii_permute[0]
    p_ii = (1+len(np.where(top_ii_permute >= real_contribution)[0]))/(1+n_permutation)
    p_val_list.append(p_ii)

# plt.figure(figsize=(8, 4))
often_top = ttl_contribute_risk_meth.nlargest(p_val_top_n, columns=['contribution'])
often_top.plot(kind='barh', fontsize=16, figsize=(8, 8))
for i, (cont, p) in enumerate(zip(often_top.iloc[:, 0].values, np.array(p_val_list))):
    plt.text(cont, i, f" p={round(p, 3)}", va="center", fontsize=16)
plt.gca().invert_yaxis()
plt.xticks(np.arange(0, 1.5, 0.5))
# ax.set_title("Top 20 closeness centrality", fontsize=16)
plt.xlabel("Total contribution", fontsize=20)
plt.ylabel("DNA Methylation", fontsize=20)
plt.tight_layout()
plt.savefig(f"top{p_val_top_n}_ttl_dna_contribute_1230_{SEX}.pdf")



# Tripartite network viz
# adjacency graph, weight is norm_MI values.
# Full network --> tripartite network
adj_mat = corr_matrix.copy()
for ii in range(len(var_names)):
    for jj in range(len(var_names)):
        var_ii = var_names[ii]
        var_jj = var_names[jj]
        if (var_ii in vars_risk_phen and var_jj in vars_risk_phen) or \
                (var_ii in vars_dna_meth and var_jj in vars_dna_meth):
            adj_mat[ii, jj] = 0


df_adj = pd.DataFrame(adj_mat, index=var_names, columns=var_names)
G = nx.from_pandas_adjacency(df_adj)
G.remove_nodes_from(list(nx.isolates(G)))
# G.remove_nodes_from(list(set(list(nx.isolates(G))) - set(var_group_phen)))
print("The number of nodes in the network is", len(G.nodes))

edges = list(G.edges)
weight = np.array([G[u][v]['weight'] for u, v in G.edges])
weight_dict = dict(zip(edges, weight))

# No risk factors
G.remove_nodes_from(exposures)
G.remove_nodes_from(list(nx.isolates(G)))
print("The number of nodes in the network is", len(G.nodes))

phen_nodes = []
meth_nodes = []
color_nodes = []
for node in G.nodes:
    if node in vars_risk_phen:
        phen_nodes.append(node)
        if node in dep_vars:
            color_nodes.append('red')
        elif node in cvd_vars:
            color_nodes.append('gold')
        else:
            print('Node label error!')
    elif node in vars_dna_meth:
        meth_nodes.append(node)
        color_nodes.append('tab:blue')
    else:
        print('Node group error!')

G_tri = nx.complete_multipartite_graph(meth_nodes, phen_nodes)
pos_tri = nx.multipartite_layout(G_tri)
node_str_degree = np.array([x[1] for x in list(G.degree)])

plt.figure(figsize=(8, 24))
nx.draw_networkx_nodes(G, pos_tri, node_size=6, node_color=color_nodes)
nx.draw_networkx_edges(G, pos_tri, width=weight, alpha=0.2)
# nx.draw_networkx_labels(G, pos=pos_tri, font_size=8, horizontalalignment='center')
plt.axis('off')
plt.tight_layout()
plt.savefig("tri_net_only_summ_p001_0823_filter_qcut.pdf")

# Gephi data preparation
nx.write_gexf(G, "bipartite_YFS_1820_MI_corr_net_p001_0729.gexf")


var_groups = [dep_vars, cvd_vars, vars_dna_meth, exposures]  # TBD
node_table = pd.read_csv("nodes_tbl_tri.csv")
gm.add_node_attribute(node_table, var_groups, 'node_group')
node_table.to_csv("nodes_tbl_tri_group.csv", index=False)

"""
plt.figure(figsize=[4, 10])
nx.draw_networkx_nodes(G, pos_tri, node_size=18, node_color=color_nodes)
nx.draw_networkx_edges(G, pos_tri, width=weight, alpha=0.2)
plt.axis('off')
plt.tight_layout()
plt.savefig("tri_net_som_cog_per20_p001_0823_filter_qcut.pdf")
"""

# projected network visualization
adj_mat_meth = project_adj_meth['level_0']
adj_mat = adj_mat_meth.copy()
df_adj = pd.DataFrame(adj_mat, index=vars_risk_phen, columns=vars_risk_phen)

for ii in range(len(adj_mat)):
    for jj in range(len(adj_mat)):
        var_ii = vars_risk_phen[ii]
        var_jj = vars_risk_phen[jj]
        if (var_ii in exposures and var_jj in exposures) or (var_ii in diseases and var_jj in diseases):
            adj_mat[ii, jj] = 0

df_adj = pd.DataFrame(adj_mat, index=vars_risk_phen, columns=vars_risk_phen)
# df_adj.to_excel("proj_2_YFS_1820_PHEN_full_adj_mat_0922_" + SEX + ".xlsx", index=True)
G = nx.from_pandas_adjacency(df_adj)
print("The number of nodes in the network is", len(G.nodes))
nx.write_gexf(G, "proj_2_YFS_1820_PHEN_full_network_0731_" + SEX + ".gexf")


# Only disease network
df_adj.drop(index=exposures, inplace=True)
df_adj.drop(columns=exposures, inplace=True)
df_adj.to_excel("proj_2_YFS_1820_PHEN_disease_adj_mat_0922_" + SEX + ".xlsx", index=True)
G = nx.from_pandas_adjacency(df_adj)
print("The number of nodes in the network is", len(G.nodes))
nx.write_gexf(G, "proj_2_YFS_1820_PHEN_disease_network_0731_" + SEX + ".gexf")


node_str_degree = np.array([x[1] for x in list(G.degree)])
# node_fun_degree = adj_mat.sum(axis=0)
# node_fun_degree = node_fun_degree / max(node_fun_degree)
edges = list(G.edges)
weight = np.array([G[u][v]['weight'] for u, v in G.edges])
weight_dict = dict(zip(edges, weight))
node_fun_degree = np.array(list(dict(G.degree(weight="weight")).values()))

# cvd_nodes = []
risk_nodes = []
# dep_nodes = []
disease_nodes = []
color_nodes = []
for node in G.nodes:
    if node in diseases:
        disease_nodes.append(node)
        color_nodes.append('#069AF3')
    elif node in exposures:
        risk_nodes.append(node)
        color_nodes.append('#FF81C0')
    else:
        print("Node label error!")

# Not yet decide the variable GROUPS, so here, it reports errors, as some variables are not in any groups.
plt.figure(figsize=(6, 6))
G_tri = nx.complete_multipartite_graph(disease_nodes, risk_nodes)
# pos_tri = nx.multipartite_layout(G_tri, align='vertical')
pos_layout = nx.spring_layout(G)
# pos_layout = pos_tri
nx.draw_networkx_nodes(G, pos=pos_layout, node_size=node_fun_degree / max(node_fun_degree) * 500, node_color=color_nodes)
nx.draw_networkx_edges(G, pos=pos_layout, width=weight / max(weight) * 5, edge_color='black',
                       alpha=0.4)  # teal and tab:gr
nx.draw_networkx_labels(G, pos=pos_layout, font_size=12, horizontalalignment='center')
plt.axis("off")
plt.tight_layout()
plt.savefig("proj_2_YFS_1820_projected_disease_network_0922_label.pdf")

contribute_exposure = df_adj.loc[exposures, diseases]
df_contribute_exposure_per = contribute_exposure.div(contribute_exposure.sum(axis=0))

# projected corr VS. MI corr
mi_corr_disease = mi_abs_disease.copy()
projected_corr_disease = adj_mat_meth.copy()

for ii in range(len(mi_corr_disease)):
    for jj in range(len(mi_corr_disease)):
        var_ii = vars_risk_phen[ii]
        var_jj = vars_risk_phen[jj]
        if (var_ii in cvd_vars and var_jj in cvd_vars) or \
                (var_ii in dep_vars and var_jj in dep_vars) or \
                (var_ii in exposures and var_jj in exposures) or \
                (var_ii in cvd_vars and var_jj in dep_vars) or \
                (var_ii in dep_vars and var_jj in cvd_vars):
            mi_corr_disease[ii, jj] = 0

for ii in range(len(projected_corr_disease)):
    for jj in range(len(projected_corr_disease)):
        var_ii = vars_risk_phen[ii]
        var_jj = vars_risk_phen[jj]
        if (var_ii in cvd_vars and var_jj in cvd_vars) or \
                (var_ii in dep_vars and var_jj in dep_vars) or \
                (var_ii in exposures and var_jj in exposures) or \
                (var_ii in cvd_vars and var_jj in dep_vars) or \
                (var_ii in dep_vars and var_jj in cvd_vars):
            projected_corr_disease[ii, jj] = 0

mi_abs_val_arr = mi_corr_disease[np.triu_indices(len(mi_corr_disease), 1)]
proj_corr_val_arr = projected_corr_disease[np.triu_indices(len(projected_corr_disease), 1)]
x = mi_abs_val_arr / max(mi_abs_val_arr)
y = proj_corr_val_arr / max(proj_corr_val_arr)
x_nonzero = x[(x != 0) & (y != 0)]
y_nonzero = y[(x != 0) & (y != 0)]
x = np.log(x_nonzero)
y = np.log(y_nonzero)
# plt.scatter(x, y, s=60, alpha=0.6)
# log_x = np.log(x)
# log_y = np.log(y)
plt.scatter(x, y, s=60, alpha=0.6)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color='k', linewidth=3, alpha=0.6)
# plt.axline((0, 0), slope=1, c='k')
plt.xlabel('log(corr)$_{MI}$', fontsize=20)
plt.ylabel('log(corr)$_{Projected}$', fontsize=20)
plt.tick_params(labelsize=18)
plt.tight_layout()
# plt.savefig("proj_2_ABS_proj_corr_vs_MI_loglog_only_sym_0328_filter_qcut_sturges_impute.pdf")

mi_corr_disease_df = pd.DataFrame(mi_corr_disease, index=vars_risk_phen, columns=vars_risk_phen)
adj_mat_meta_lipid_df = pd.DataFrame(projected_corr_disease, index=vars_risk_phen, columns=vars_risk_phen)

plt.figure(figsize=(20, 11.5))
ax = sns.heatmap(mi_corr_disease, cmap='Spectral_r', xticklabels=vars_risk_phen, yticklabels=vars_risk_phen,
                 square=True)
ax.tick_params(labelsize=18)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
# plt.savefig("norm_MI_phen_matrix_only_sym_0328_filtering_sturges_impute.pdf")

plt.figure(figsize=(20, 11.5))
ax = sns.heatmap(projected_corr_disease, cmap='Spectral_r', xticklabels=vars_risk_phen, yticklabels=vars_risk_phen,
                 square=True)
ax.tick_params(labelsize=18)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig("proj_phen_matrix_only_sym_0328_filtering_sturges_impute.pdf")

# Linear comparison between MI and projected correlation.
mi_abs_val_arr = mi_corr_disease[np.triu_indices(len(mi_corr_disease), 1)]
proj_corr_val_arr = projected_corr_disease[np.triu_indices(len(projected_corr_disease), 1)]
plt.scatter(mi_abs_val_arr / max(mi_abs_val_arr), proj_corr_val_arr / max(proj_corr_val_arr), s=60, alpha=0.6)
plt.xlabel('MI correlation', fontsize=20)
plt.ylabel('Projected correlation', fontsize=20)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig("proj_2_corr_vs_MI_tri_only_sym_0328_filter_qcut_sturges_impute.pdf")

# Functional degree comparison
node_fun_degree_mi = mi_corr_disease.sum(axis=0)
node_fun_degree_proj = projected_corr_disease.sum(axis=0)
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
