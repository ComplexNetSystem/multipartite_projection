import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
import matplotlib.pyplot as plt
import random
from itertools import combinations
from itertools import product


def cluster_to_tbl(community_partition, csv_name):
    var_name = np.array(list(community_partition.keys()))
    var_cluster = list(community_partition.values())
    clusters = np.unique(var_cluster)
    writer = pd.ExcelWriter(csv_name, engine="openpyxl")
    for c_index in clusters:
        indices = [i for i, x in enumerate(var_cluster) if x == c_index]
        c_vars = var_name[indices]
        df_cluster = pd.DataFrame({"index": indices, "cluster-" + str(c_index): c_vars})
        df_cluster.to_excel(writer, sheet_name="cluster-" + str(c_index))

    writer.save()


def cluster_to_dict(community_partition):
    # var_name = np.array(list(community_partition.keys()))
    var_cluster = list(community_partition.values())
    clusters = np.unique(var_cluster)
    cluster_dict = {}
    for c_index in clusters:
        indices = [i for i, x in enumerate(var_cluster) if x == c_index]
        cluster_dict[str(c_index)] = indices

    return cluster_dict


def o_info_comb_in_cluster(cluster_dict, n_set, csv_name):
    clusters = list(cluster_dict.keys())
    comb_in_cluster = []
    for c_num in clusters:
        index_in_cluster = cluster_dict[c_num]
        comb_in_cluster.extend(combinations(index_in_cluster, n_set))
    comb_in_cluster_df = pd.DataFrame({'comb': comb_in_cluster})
    comb_in_cluster_df.to_csv(csv_name)


def o_info_comb_btn3_cluster(cluster_dict, n_set, csv_name):
    clusters = list(cluster_dict.keys())
    comb_cluster_all = list(combinations(clusters, n_set))
    comb_btn3_cluster = []
    for comb_cluster_num in comb_cluster_all:
        index_in_clusters = []
        for c_num in comb_cluster_num:
            index_in_clusters.append(cluster_dict[c_num])
        comb_btn3_cluster.extend(list(product(*index_in_clusters)))
    sorted_comb_btn3_cluster = [tuple(sorted(ii)) for ii in comb_btn3_cluster]
    sorted_comb_btn3_cluster_df = pd.DataFrame({'comb': sorted_comb_btn3_cluster})
    sorted_comb_btn3_cluster_df.to_csv(csv_name)


def net_mat_threshold_old(mat, top_per, is_directed=False):
    if is_directed:
        mat_sorted = sorted(np.reshape(mat, mat.size), reverse=True)  # directed graph
    else:
        mat_sorted = sorted(mat[np.triu_indices(len(mat), 1)], reverse=True)  # undirected graph
    mat_sorted = [i for i in mat_sorted if i != 0]  # Remove all zeros from the list.
    threshold = mat_sorted[round(len(mat_sorted) * top_per) - 1]
    return threshold


def net_mat_threshold(mat, top_per, is_directed=False):
    if is_directed:
        mat_sorted = sorted(np.reshape(mat, mat.size), reverse=True)  # directed graph
    else:
        mat_sorted = sorted(mat[np.triu_indices(len(mat), 1)], reverse=True)  # undirected graph
    # Remove all zeros and nan from the list.
    mat_sorted = sorted([i for i in mat_sorted if i != 0 and not np.isnan(i)], reverse=True)
    threshold = mat_sorted[round(len(mat_sorted) * top_per) - 1]
    return threshold


def list2tuple_in_list(list_in_list):
    tuple_in_list = []
    for list_ele in list_in_list:
        tuple_in_list.append(tuple(list_ele))
    return tuple_in_list


def cond_ent_mat(mi, min_mean_directed):
    num_var = len(mi)
    cond_ent_matrix = np.zeros([num_var, num_var])
    for ii in range(num_var):
        for jj in range(num_var):
            # H(X|Y) = H(X) - I(X;Y); H(Y|X) = H(Y) - I(X;Y)
            if ii < jj:
                cond_ent_1 = mi[ii, ii] - mi[ii, jj]
                cond_ent_2 = mi[jj, jj] - mi[ii, jj]
            else:
                continue

            if min_mean_directed == 'min':
                cond_ent_matrix[ii, jj] = np.min([cond_ent_1, cond_ent_2])
                cond_ent_matrix[jj, ii] = np.min([cond_ent_1, cond_ent_2])
            elif min_mean_directed == 'mean':
                cond_ent_matrix[ii, jj] = np.mean([cond_ent_1, cond_ent_2])
                cond_ent_matrix[jj, ii] = np.mean([cond_ent_1, cond_ent_2])
            elif min_mean_directed == 'directed':
                cond_ent_matrix[ii, jj] = cond_ent_2
                cond_ent_matrix[jj, ii] = cond_ent_1

    return cond_ent_matrix


def norm_cond_ent_mat(mi, min_max_mean_directed):
    num_var = len(mi)
    norm_cond_ent_matrix = np.zeros([num_var, num_var])
    for ii in range(num_var):
        for jj in range(num_var):
            # H(X|Y) = H(X) - I(X;Y); H(Y|X) = H(Y) - I(X;Y)
            norm_factor = np.max([mi[ii, ii], mi[jj, jj]])
            if ii < jj:
                norm_cond_ent_1 = (mi[ii, ii] - mi[ii, jj]) / norm_factor
                norm_cond_ent_2 = (mi[jj, jj] - mi[ii, jj]) / norm_factor
            else:
                continue

            if min_max_mean_directed == 'min':
                norm_cond_ent_matrix[ii, jj] = np.min([norm_cond_ent_1, norm_cond_ent_2])
                norm_cond_ent_matrix[jj, ii] = np.min([norm_cond_ent_1, norm_cond_ent_2])
            elif min_max_mean_directed == 'max':
                norm_cond_ent_matrix[ii, jj] = np.max([norm_cond_ent_1, norm_cond_ent_2])
                norm_cond_ent_matrix[jj, ii] = np.max([norm_cond_ent_1, norm_cond_ent_2])
            elif min_max_mean_directed == 'mean':
                norm_cond_ent_matrix[ii, jj] = np.mean([norm_cond_ent_1, norm_cond_ent_2])
                norm_cond_ent_matrix[jj, ii] = np.mean([norm_cond_ent_1, norm_cond_ent_2])
            elif min_max_mean_directed == 'directed':
                norm_cond_ent_matrix[ii, jj] = norm_cond_ent_2
                norm_cond_ent_matrix[jj, ii] = norm_cond_ent_1

    return norm_cond_ent_matrix


def directed_undirected(directed_mat, min_mean):
    num_var = len(directed_mat)
    undirected_matrix = np.zeros([num_var, num_var])
    for ii in range(num_var):
        for jj in range(num_var):
            if ii < jj:
                ii_jj = directed_mat[ii, jj]
                jj_ii = directed_mat[jj, ii]
                if ii_jj == 0:
                    undirected_matrix[ii, jj] = jj_ii
                elif jj_ii == 0:
                    undirected_matrix[jj, ii] = ii_jj
                else:
                    if min_mean == 'min':
                        undirected_matrix[ii, jj] = np.min([ii_jj, jj_ii])
                        undirected_matrix[jj, ii] = np.min([ii_jj, jj_ii])
                    elif min_mean == 'mean':
                        undirected_matrix[ii, jj] = np.mean([ii_jj, jj_ii])
                        undirected_matrix[jj, ii] = np.mean([ii_jj, jj_ii])
            else:
                continue

    return undirected_matrix


def detected_percent(mi_mat, top_range):
    start = top_range[0]
    end = top_range[1]
    step_len = 0.001
    step_num = int((end - start) / step_len + 1)
    num_detected_triangle = []
    num_triangle_in_top_synergy = []
    detected_per = []
    num_random_tri_in_top_synergy = []
    random_per = []
    o_info_tri = pd.read_csv("o_info_triplets.csv", index_col=0)
    for top in np.linspace(start, end, num=step_num):
        print(top)
        adj_mat = cond_ent_mat(mi_mat, min_mean_directed="mean")  # MI matrix -> CH undirected adj matrix
        threshold = net_mat_threshold(adj_mat, top)
        adj_mat[adj_mat < threshold] = 0

        G = nx.from_numpy_matrix(adj_mat)
        all_cliques = list(nx.enumerate_all_cliques(G))
        triadic_cliques = [clq for clq in all_cliques if len(clq) == 3]
        num_detected_triangle.append(len(triadic_cliques))

        # Format the triadic cliques: [1, 2, 3] -> (1, 2, 3) -> '(1, 2, 3)'
        triadic_cliques_tuple = list2tuple_in_list(triadic_cliques)
        tri_clq_df = pd.DataFrame({'comb': triadic_cliques_tuple})
        tri_clq_df.to_csv("triadic_cliques.csv")
        triadic_cliques_df = pd.read_csv("triadic_cliques.csv", index_col=0)

        o_info_tri_clq = o_info_tri.loc[o_info_tri['Combinations'].isin(triadic_cliques_df['comb'])]
        random_tri = random.sample(list(o_info_tri['Combinations']), len(o_info_tri_clq))
        o_info_random_tri = o_info_tri.loc[o_info_tri['Combinations'].isin(random_tri)]
        synergy_top = o_info_tri.nsmallest(len(o_info_tri_clq), 'O-info')
        # synergy_top = o_info_tri.sort_values(by='O-info', ascending=False).tail(len(o_info_tri_clq))

        # Calculate the overlap of triplets
        idx_o_info_clq = pd.Index(o_info_tri_clq['Combinations'])
        idx_o_info_random = pd.Index(o_info_random_tri['Combinations'])
        idx_synergy_top = pd.Index(synergy_top['Combinations'])

        common_triplet_clq = idx_synergy_top.intersection(idx_o_info_clq)
        num_triangle_in_top_synergy.append(len(common_triplet_clq))
        detected_per.append(len(common_triplet_clq) / len(idx_synergy_top))

        common_triplet_random = idx_synergy_top.intersection(idx_o_info_random)
        num_random_tri_in_top_synergy.append(len(common_triplet_random))
        random_per.append(len(common_triplet_random) / len(idx_synergy_top))

    diff_step = list(np.diff(num_triangle_in_top_synergy))
    diff_step.insert(0, num_triangle_in_top_synergy[0])
    df_top_detected_percent = pd.DataFrame({'top': np.linspace(start, end, num=step_num),
                                            'detected_triangle': num_detected_triangle,
                                            'triangle_in_top_synergy': num_triangle_in_top_synergy,
                                            'diff_detected_top_tri': diff_step,
                                            'detect_percent': detected_per,
                                            'random_tri_in_top_synergy': num_random_tri_in_top_synergy,
                                            'random_percent': random_per})
    return df_top_detected_percent


def detected_synergy_top_n(cond_ent_range, cond_ent_in_cluster, idx_o_info_clq, synergy_top, o_info_tri, min_mean,
                           top_n):
    start = top_n[0]
    end = top_n[1]
    step_len = 10
    step_num = int((end - start) / step_len + 1)

    num_detected_top_synergy = []
    per_detected_top_synergy = []

    if min_mean == 'min':
        cond_ent_in_cluster_range = cond_ent_in_cluster.loc[(cond_ent_in_cluster['min_cond_ent'] >= cond_ent_range)]
    elif min_mean == 'mean':
        cond_ent_in_cluster_range = cond_ent_in_cluster.loc[(cond_ent_in_cluster['mean_cond_ent'] >= cond_ent_range)]

    idx_ch_tri_range_in_cluster = pd.Index(cond_ent_in_cluster_range['Combinations'])
    union_tri = idx_o_info_clq.union(idx_ch_tri_range_in_cluster)
    print("The total number of identified triplets: ", len(union_tri))

    idx_synergy_top = pd.Index(synergy_top['Combinations'])
    common_triplet = idx_synergy_top.intersection(union_tri)
    print("The total number of identified triplets in top synergy: ", len(common_triplet))

    for syn_top in np.linspace(start, end, num=step_num):
        synergy_top_example = o_info_tri.nsmallest(int(syn_top), 'O-info')
        idx_synergy_top_example = pd.Index(synergy_top_example['Combinations'])
        common_synergy_top_n = len(common_triplet.intersection(idx_synergy_top_example))
        num_detected_top_synergy.append(common_synergy_top_n)
        per_detected_top_synergy.append(common_synergy_top_n / syn_top)

    df_top_detected = pd.DataFrame({'top_n': np.linspace(start, end, num=step_num),
                                    'detected_top_synergy': num_detected_top_synergy,
                                    'detected_top_synergy_per': per_detected_top_synergy})
    return df_top_detected


def found_syn_ch_order(found_tri_syn, found_tri_ch, all_tri_syn, top_n):
    common_num = []
    common_ratio = []
    common_num_all = []
    common_ratio_all = []
    step_n = 1

    all_syn_tri_rank = all_tri_syn.sort_values(by='O-info')

    for top_num in range(1, top_n + 1, step_n):
        detected_synergy_top = found_tri_syn.nsmallest(top_num, 'O-info')
        all_synergy_top = all_syn_tri_rank.head(top_num)
        # all_synergy_top = all_tri_syn.nsmallest(top_num, 'O-info')
        ch_top = found_tri_ch.nlargest(top_num, 'mean_CH_tri')
        idx_detected_synergy_top = pd.Index(detected_synergy_top['Combinations'])
        idx_all_synergy_top = pd.Index(all_synergy_top['Combinations'])
        idx_ch_top = pd.Index(ch_top['Combinations'])
        common_detected_syn_ch_top_n = len(idx_detected_synergy_top.intersection(idx_ch_top))
        common_num.append(common_detected_syn_ch_top_n)
        common_ratio.append(common_detected_syn_ch_top_n / top_num)
        common_all_syn_ch_top_n = len(idx_all_synergy_top.intersection(idx_ch_top))
        common_num_all.append(common_all_syn_ch_top_n)
        common_ratio_all.append(common_all_syn_ch_top_n / top_num)

    df_order_ratio = pd.DataFrame({'top_n': range(1, top_n + 1, step_n),
                                   'common_num_vs_detected_tri': common_num,
                                   'common_ratio_vs_detected_tri': common_ratio,
                                   'common_num_vs_all_tri': common_num_all,
                                   'common_ratio_vs_all_tri': common_ratio_all})
    return df_order_ratio


def degree_dist_ent(adj_mat, threshold_percent):
    degree_ent = []
    for top_per in threshold_percent:
        net_mat = np.copy(adj_mat)
        thr_val = net_mat_threshold(net_mat, top_per)
        net_mat[net_mat < thr_val] = 0
        df_adj = pd.DataFrame(net_mat)
        G = nx.from_pandas_adjacency(df_adj)
        G.remove_nodes_from(list(nx.isolates(G)))
        node_str_degree = np.array([x[1] for x in list(G.degree)])
        # unique, cnt = np.unique(node_str_degree, return_counts=True)
        # node_fun_degree = adj_mat.sum(axis=0)
        num_var = len(G.nodes)
        ent_min = np.log(4 * (num_var - 1)) / 2
        ent_max = np.log(num_var)

        ent_unit_node = []
        for node_degree_val in node_str_degree / np.sum(node_str_degree):
            ent_net_node = -1 * node_degree_val * np.log(node_degree_val)
            ent_unit_node.append(ent_net_node)
        net_ent = sum(ent_unit_node)
        degree_ent.append((net_ent - ent_min) / (ent_max - ent_min))

    return degree_ent


# Just wrote the function to test, but the performance
# this functional degree distribution Shannon entropy is not good!!
def func_degree_dist_ent(adj_mat, threshold_percent):
    func_degree_ent = []
    for top_per in threshold_percent:
        net_mat = np.copy(adj_mat)
        thr_val = net_mat_threshold(net_mat, top_per)
        net_mat[net_mat < thr_val] = 0
        df_adj = pd.DataFrame(net_mat)
        G = nx.from_pandas_adjacency(df_adj)
        G.remove_nodes_from(list(nx.isolates(G)))
        node_fun_degree = adj_mat.sum(axis=0)
        node_fun_degree[node_fun_degree != 0]
        num_var = len(G.nodes)
        ent_min = np.log(4 * (num_var - 1)) / 2
        ent_max = np.log(num_var)
        ent_unit_node = []
        for node_degree_val in node_fun_degree / np.sum(node_fun_degree):
            ent_net_node = -1 * node_degree_val * np.log(node_degree_val)
            ent_unit_node.append(ent_net_node)
        net_ent = sum(ent_unit_node)
        func_degree_ent.append((net_ent - ent_min) / (ent_max - ent_min))
    return func_degree_ent


def dist_reciprocal_weight(w_mat):
    num = len(w_mat)
    dist_mat = np.zeros([num, num])
    for ii in range(num):
        for jj in range(num):
            if w_mat[ii, jj] == 0:
                continue
            else:
                dist_mat[ii, jj] = 1 / w_mat[ii, jj]
    return dist_mat


def log_10_weight(w_mat):
    num = len(w_mat)
    log_mat = np.zeros([num, num])
    for ii in range(num):
        for jj in range(num):
            if w_mat[ii, jj] == 0:
                continue
            else:
                log_mat[ii, jj] = np.log10(w_mat[ii, jj])
    return log_mat


def sal_mat_from_all_spt(spt, num_node):
    sal_mat = np.zeros([num_node, num_node])
    for ref_var in spt.keys():
        t_mat_ref = np.zeros([num_node, num_node])
        for dst_var in spt[ref_var].keys():
            path = spt[ref_var][dst_var]
            for kk in range(len(path) - 1):
                path_src_node = path[kk]
                path_dst_node = path[kk + 1]
                t_mat_ref[path_src_node, path_dst_node] = 1
                t_mat_ref[path_dst_node, path_src_node] = 1  # undirected graph
        sal_mat += t_mat_ref
    sal_mat = sal_mat / num_node
    return sal_mat


def sal_mat_from_all_spt_name(spt, var_name_list):
    num_node = len(var_name_list)
    sal_mat = np.zeros([num_node, num_node])
    for ref_var in spt.keys():
        t_mat_ref = np.zeros([num_node, num_node])
        for dst_var in spt[ref_var].keys():
            path = spt[ref_var][dst_var]
            for kk in range(len(path) - 1):
                path_src_node = path[kk]
                path_src_idx = var_name_list.index(path_src_node)
                path_dst_node = path[kk + 1]
                path_dst_idx = var_name_list.index(path_dst_node)
                t_mat_ref[path_src_idx, path_dst_idx] = 1
                t_mat_ref[path_dst_idx, path_src_idx] = 1  # undirected graph
        sal_mat += t_mat_ref
    sal_mat = sal_mat / num_node
    return sal_mat


def edge_centrality_mat(edge_cen_dist, var_name_list):
    num_node = len(var_name_list)
    centrality_mat = np.zeros([num_node, num_node])
    for edge in edge_cen_dist.keys():
        edge_src_node = edge[0]
        edge_src_idx = var_name_list.index(edge_src_node)
        edge_dst_node = edge[1]
        edge_dst_idx = var_name_list.index(edge_dst_node)
        centrality_mat[edge_src_idx, edge_dst_idx] = edge_cen_dist[edge]
        centrality_mat[edge_dst_idx, edge_src_idx] = edge_cen_dist[edge]

    return centrality_mat


def sym_mat_flat(sym_mat):
    num_var = sym_mat.shape[0]
    flatten_list = []
    for ii in range(num_var):
        for jj in range(num_var):
            if ii < jj:
                flatten_list.append(sym_mat[ii, jj])
            else:
                continue
    return flatten_list


def power_law_plot(data_arr, x_label):
    # The latest one, updated on May 10, 2024.
    data_nonzero = data_arr[data_arr != 0]
    fit_all = powerlaw.Fit(data_nonzero, xmin=min(data_nonzero))  # no x_min -> x_min is the minimum
    x_all, y_all = fit_all.ccdf()
    fit = powerlaw.Fit(data_nonzero)  # auto select x_min
    # x, y = fit.ccdf()   # Unique values of fit.data and actual CCDF.
    x_fit_data = fit.data    # Original data greater than x_min.
    y_fit_value = fit.power_law.ccdf()   # Theoretical fitted CCDF for fit.data

    # power-law plot only from x_min
    # fig1 = fit.plot_ccdf(linestyle='none', marker='o', markerfacecolor='none')
    # fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig1)

    # Adaptation for the power-law plot of all data
    fit_all.plot_ccdf(linestyle='none', marker='o', markerfacecolor='none', label='P(X)')

    # Theoretical CCDF(fit.data higher than x_min) - CCDF(x_min in all original data)  --> CCDF considering all data.
    # Note that this is in Log-Log scale. Minus should be division. Inversely, should be multiplication.
    plt.loglog(x_fit_data, y_fit_value * y_all[x_all == fit.xmin], color='r', linestyle='--',
               label=r'$-\alpha+1$=' + str(round(1 - fit.alpha, 2)))
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(r"P(X$\geq$x)", fontsize=14)
    plt.legend()

    return [fit.xmin, fit.alpha]


def shared_neighbor_find(graph, var_list, n_set):
    vars_neighbor_set = []
    [vars_neighbor_set.append(set(graph.neighbors(var))) for var in var_list]
    var_combs = list(combinations(range(len(var_list)), n_set))
    shared_neighbor_all_combs = set()
    neighbor_link_all_combs = set()
    for var_comb in var_combs:
        vars_neighbor_set_list = list(np.array(vars_neighbor_set)[list(var_comb)])
        shared_neighbor_per_comb = set.intersection(*vars_neighbor_set_list)
        shared_neighbor_all_combs = shared_neighbor_all_combs.union(shared_neighbor_per_comb)

    return shared_neighbor_all_combs


def shared_neighbor_link_find(graph, var_list, n_set, counted_vars):
    vars_neighbor_set = []
    [vars_neighbor_set.append(set(graph.neighbors(var))) for var in var_list]
    var_combs = list(combinations(range(len(var_list)), n_set))
    shared_neighbor_all_combs = set()
    neighbor_link_all_combs = set()
    for var_comb in var_combs:
        vars_in_comb = list(np.array(var_list)[list(var_comb)])
        vars_neighbor_set_list = list(np.array(vars_neighbor_set)[list(var_comb)])
        shared_neighbor_per_comb = set.intersection(*vars_neighbor_set_list)
        shared_neighbor_per_comb.difference_update(counted_vars)
        for neighbor in shared_neighbor_per_comb:
            [neighbor_link_all_combs.add(tuple(set((var_in_comb, neighbor)))) for var_in_comb in vars_in_comb]
        shared_neighbor_all_combs = shared_neighbor_all_combs.union(shared_neighbor_per_comb)

    return shared_neighbor_all_combs, neighbor_link_all_combs


def shared_neighbor_tree(graph, var_list, n_set):
    var_set = set(var_list)
    all_counted_vars_current = var_set
    all_counted_links_current = set()
    num_level = 0
    shared_neighbor_level = {'level_' + str(num_level): var_list}
    neighbor_links_level = {'level_' + str(num_level): set()}
    while len(var_list) > 1:
        num_level = num_level + 1
        if n_set == 'all' or n_set > len(var_list):
            shared_neighbors, neighbor_links = shared_neighbor_link_find(graph, var_list,
                                                                         len(var_list), all_counted_vars_current)
        else:
            shared_neighbors, neighbor_links = shared_neighbor_link_find(graph, var_list,
                                                                         n_set, all_counted_vars_current)
        shared_neighbors.difference_update(all_counted_vars_current)
        # update var_set
        var_list = list(shared_neighbors)
        shared_neighbor_level['level_' + str(num_level)] = var_list
        neighbor_links.difference_update(all_counted_links_current)
        neighbor_links_level['level_' + str(num_level)] = neighbor_links
        # Dangerous: DO NOT use update like this: all_counted_vars_current.update(shared_neighbors)
        all_counted_vars_current = all_counted_vars_current.union(shared_neighbors)
        all_counted_links_current = all_counted_links_current.union(neighbor_links)

    return shared_neighbor_level, neighbor_links_level


def neighbor_tree_adj(original_adj, tree_links_dict, var_names):
    neighbor_tree_adj_mat = np.zeros(original_adj.shape)
    for link_set in tree_links_dict.values():
        if link_set != set():
            for link in link_set:
                link_node_src = link[0]
                link_node_end = link[1]
                src_index = np.where(var_names == link_node_src)
                end_index = np.where(var_names == link_node_end)
                neighbor_tree_adj_mat[src_index, end_index] = original_adj[src_index, end_index]
                neighbor_tree_adj_mat[end_index, src_index] = original_adj[end_index, src_index]

    return neighbor_tree_adj_mat


def bipartite_adj_func(original_adj, links_set, var_names):
    bipartite_adj_mat = np.zeros(original_adj.shape)
    for link in links_set:
        link_node_src = link[0]
        link_node_end = link[1]
        src_index = np.where(var_names == link_node_src)
        end_index = np.where(var_names == link_node_end)
        bipartite_adj_mat[src_index, end_index] = original_adj[src_index, end_index]
        bipartite_adj_mat[end_index, src_index] = original_adj[end_index, src_index]
    return bipartite_adj_mat


def joint_score_cal(df_adj: pd.DataFrame, cvd: list, dep: list, biomarkers: list):
    bio_joint_score = {}
    for biomarker in biomarkers:
        corr_bio_cvd = df_adj[biomarker].loc[cvd].sum()
        corr_bio_dep = df_adj[biomarker].loc[dep].sum()
        bio_joint_score[biomarker] = corr_bio_cvd * corr_bio_dep

    bio_joint_score_df = pd.DataFrame.from_dict(bio_joint_score, orient='index', columns=['score'])
    bio_joint_score_df.sort_values(by=['score'], ascending=False, inplace=True)
    return bio_joint_score_df


def adj_within_level(G, level_group_dict, n_set):
    level_num = 0
    counted_vars = set()
    adj_all_levels = {}
    for level_vars in level_group_dict.values():
        counted_vars = counted_vars.union(set(level_vars))
        num_var_level = len(level_vars)
        adj_level = np.zeros((num_var_level, num_var_level))
        var_neighbors_level = []
        [var_neighbors_level.append(set(G.neighbors(var))) for var in level_vars]
        var_combs_level = list(combinations(range(len(level_vars)), n_set))
        for var_comb in var_combs_level:
            # vars_in_comb = list(np.array(level_vars)[list(var_comb)])
            vars_neighbor_set_list = list(np.array(var_neighbors_level)[list(var_comb)])
            shared_neighbor_per_comb = set.intersection(*vars_neighbor_set_list)
            shared_neighbor_per_comb.difference_update(counted_vars)
            adj_level[var_comb[0], var_comb[1]] = len(shared_neighbor_per_comb)
            adj_level[var_comb[1], var_comb[0]] = len(shared_neighbor_per_comb)

        adj_all_levels['level_' + str(level_num)] = adj_level
        level_num = level_num + 1
    return adj_all_levels


def adj_within_level_shared_neighbor(G, level_group_dict, n_set):
    level_num = 0
    counted_vars = set()
    adj_all_levels = {}
    for level_vars in level_group_dict.values():
        counted_vars = counted_vars.union(set(level_vars))
        num_var_level = len(level_vars)
        adj_level = np.zeros((num_var_level, num_var_level))
        var_combs_level = list(combinations(range(len(level_vars)), n_set))
        for var_comb in var_combs_level:
            src_node_idx = var_comb[0]
            dst_node_idx = var_comb[1]
            shared_neighbor = set(nx.common_neighbors(G, level_vars[src_node_idx], level_vars[dst_node_idx]))
            shared_neighbor.difference_update(counted_vars)
            adj_level[src_node_idx, dst_node_idx] = len(shared_neighbor)
            adj_level[dst_node_idx, src_node_idx] = len(shared_neighbor)

        adj_all_levels['level_' + str(level_num)] = adj_level
        level_num = level_num + 1
    return adj_all_levels


def adj_within_level_shared_neighbor_weighted(G, ori_adj, var_names, level_group_dict, n_set):
    level_num = 0
    counted_vars = set()
    adj_all_levels = {}
    for level_vars in level_group_dict.values():
        counted_vars = counted_vars.union(set(level_vars))
        num_var_level = len(level_vars)
        adj_level = np.zeros((num_var_level, num_var_level))
        var_combs_level = list(combinations(range(len(level_vars)), n_set))
        for var_comb in var_combs_level:
            src_node_idx = var_comb[0]
            dst_node_idx = var_comb[1]
            shared_neighbor = set(nx.common_neighbors(G, level_vars[src_node_idx], level_vars[dst_node_idx]))
            shared_neighbor.difference_update(counted_vars)

            src_idx_in_original = np.where(var_names == level_vars[src_node_idx])
            dst_idx_in_original = np.where(var_names == level_vars[dst_node_idx])
            shared_neighbor_weight = 0
            if shared_neighbor:
                for nei in shared_neighbor:
                    nei_idx_in_original = np.where(var_names == nei)  # nei: neighbor
                    nei_weight = ori_adj[src_idx_in_original, nei_idx_in_original] + \
                                 ori_adj[dst_idx_in_original, nei_idx_in_original]
                    shared_neighbor_weight = shared_neighbor_weight + nei_weight / 2

            adj_level[src_node_idx, dst_node_idx] = shared_neighbor_weight
            adj_level[dst_node_idx, src_node_idx] = shared_neighbor_weight

        adj_all_levels['level_' + str(level_num)] = adj_level
        level_num = level_num + 1
    return adj_all_levels


def adj_within_level_shared_neighbor_weighted_centrality(G, ori_adj, centrality, var_names, level_group_dict, n_set):
    level_num = 0
    counted_vars = set()
    adj_all_levels = {}
    for level_vars in level_group_dict.values():
        counted_vars = counted_vars.union(set(level_vars))
        num_var_level = len(level_vars)
        adj_level = np.zeros((num_var_level, num_var_level))
        var_combs_level = list(combinations(range(len(level_vars)), n_set))
        for var_comb in var_combs_level:
            src_node_idx = var_comb[0]
            dst_node_idx = var_comb[1]
            shared_neighbor = set(nx.common_neighbors(G, level_vars[src_node_idx], level_vars[dst_node_idx]))
            shared_neighbor.difference_update(counted_vars)

            src_idx_in_original = np.where(var_names == level_vars[src_node_idx])
            dst_idx_in_original = np.where(var_names == level_vars[dst_node_idx])
            shared_neighbor_weight = 0
            if shared_neighbor:
                for nei in shared_neighbor:
                    nei_idx_in_original = np.where(var_names == nei)  # nei: neighbor
                    key_cen_link_src = (level_vars[src_node_idx], nei)
                    key_cen_link_dst = (level_vars[dst_node_idx], nei)
                    if key_cen_link_src not in centrality.keys():
                        key_cen_link_src = (nei, level_vars[src_node_idx])
                    if key_cen_link_dst not in centrality.keys():
                        key_cen_link_dst = (nei, level_vars[dst_node_idx])

                    edge_cen_src = centrality[key_cen_link_src]
                    edge_cen_dst = centrality[key_cen_link_dst]
                    nei_weight = ori_adj[src_idx_in_original, nei_idx_in_original] * edge_cen_src + \
                                 ori_adj[dst_idx_in_original, nei_idx_in_original] * edge_cen_dst
                    shared_neighbor_weight = shared_neighbor_weight + nei_weight

            adj_level[src_node_idx, dst_node_idx] = shared_neighbor_weight
            adj_level[dst_node_idx, src_node_idx] = shared_neighbor_weight

        adj_all_levels['level_' + str(level_num)] = adj_level
        level_num = level_num + 1
    return adj_all_levels


def adj_within_level_multiplex(G, next_dist_G, ori_adj, var_names, level_group_dict, n_set):
    level_num = 0
    counted_vars = set()
    adj_all_levels = {}
    for level_vars in level_group_dict.values():
        counted_vars = counted_vars.union(set(level_vars))
        num_var_level = len(level_vars)
        adj_level = np.zeros((num_var_level, num_var_level))
        var_combs_level = list(combinations(range(num_var_level), n_set))
        for var_comb in var_combs_level:
            print(var_comb)
            src_node_idx = var_comb[0]
            src_idx_in_original = np.where(var_names == level_vars[src_node_idx])
            neighbor_src = set(G.neighbors(level_vars[src_node_idx]))
            neighbor_src.difference_update(counted_vars)

            dst_node_idx = var_comb[1]
            dst_idx_in_original = np.where(var_names == level_vars[dst_node_idx])
            neighbor_dst = set(G.neighbors(level_vars[dst_node_idx]))
            neighbor_dst.difference_update(counted_vars)
            # link_neighbor_src_dst = set(product(*[neighbor_src, neighbor_dst]))
            weight_multiplex_via_neighbor = 0
            for nei_src in list(neighbor_src):
                nei_src_idx = np.where(var_names == nei_src)
                weight_src_nei = ori_adj[src_idx_in_original, nei_src_idx]
                for nei_dst in list(neighbor_dst):
                    if nei_src != nei_dst and nx.has_path(next_dist_G, source=nei_src, target=nei_dst):
                        nei_dst_idx = np.where(var_names == nei_dst)
                        weight_dst_nei = ori_adj[dst_idx_in_original, nei_dst_idx]
                        sp_len = nx.shortest_path_length(next_dist_G, source=nei_src, target=nei_dst, weight='weight')
                        nei_weight = (weight_src_nei + weight_dst_nei) / sp_len
                        weight_multiplex_via_neighbor = weight_multiplex_via_neighbor + nei_weight
                    else:
                        continue

            adj_level[src_node_idx, dst_node_idx] = weight_multiplex_via_neighbor
            adj_level[dst_node_idx, src_node_idx] = weight_multiplex_via_neighbor

        adj_all_levels['level_' + str(level_num)] = adj_level
        level_num = level_num + 1
    return adj_all_levels


def adj_within_level_multiplex_exp(G, next_dist_G, ori_adj, var_names, level_group_dict, n_set):
    level_num = 0
    counted_vars = set()
    adj_all_levels = {}
    for level_vars in level_group_dict.values():
        counted_vars = counted_vars.union(set(level_vars))
        num_var_level = len(level_vars)
        adj_level = np.zeros((num_var_level, num_var_level))
        var_combs_level = list(combinations(range(num_var_level), n_set))
        for var_comb in var_combs_level:
            print(var_comb)
            src_node_idx = var_comb[0]
            src_idx_in_original = np.where(var_names == level_vars[src_node_idx])
            neighbor_src = set(G.neighbors(level_vars[src_node_idx]))
            neighbor_src.difference_update(counted_vars)

            dst_node_idx = var_comb[1]
            dst_idx_in_original = np.where(var_names == level_vars[dst_node_idx])
            neighbor_dst = set(G.neighbors(level_vars[dst_node_idx]))
            neighbor_dst.difference_update(counted_vars)
            # link_neighbor_src_dst = set(product(*[neighbor_src, neighbor_dst]))
            weight_multiplex_via_neighbor = 0
            for nei_src in list(neighbor_src):
                nei_src_idx = np.where(var_names == nei_src)
                weight_src_nei = ori_adj[src_idx_in_original, nei_src_idx]
                for nei_dst in list(neighbor_dst):
                    if nx.has_path(next_dist_G, source=nei_src, target=nei_dst):
                        nei_dst_idx = np.where(var_names == nei_dst)
                        weight_dst_nei = ori_adj[dst_idx_in_original, nei_dst_idx]
                        sp_len = nx.shortest_path_length(next_dist_G, source=nei_src, target=nei_dst, weight='weight')
                        nei_weight = (weight_src_nei + weight_dst_nei) * np.exp(-sp_len)
                        weight_multiplex_via_neighbor = weight_multiplex_via_neighbor + nei_weight
                    else:
                        continue

            adj_level[src_node_idx, dst_node_idx] = weight_multiplex_via_neighbor
            adj_level[dst_node_idx, src_node_idx] = weight_multiplex_via_neighbor

        adj_all_levels['level_' + str(level_num)] = adj_level
        level_num = level_num + 1
    return adj_all_levels


def hierarchy_pos_vertical(node_level_dict, x_diff_max_node_set, y_diff):
    num_level = len(node_level_dict)
    pos = {}
    level_s = 0
    num_node_level = [len(level_node) for level_node in node_level_dict.values()]
    max_num_node_level = max(num_node_level)
    max_x = max_num_node_level * x_diff_max_node_set  # no need to use "max_num_node_level-1"
    mid_x = max_x / 2
    for node_list in node_level_dict.values():
        num_node = len(node_list)
        if num_node == 0:
            print("Network ends at this level.")
        elif num_node == 1:
            pos[node_list[0]] = (mid_x, y_diff * (num_level - level_s))
        elif num_node > 1:
            mid_node_index = np.ceil(num_node / 2)
            x_level_diff = max_x / (num_node - 1)
            node_s = 1
            if (num_node % 2) == 1:
                for node in node_list:
                    if node_s < mid_node_index:
                        pos[node] = (mid_x - x_level_diff * (mid_node_index - node_s),
                                     y_diff * (num_level - level_s))
                    elif node_s >= mid_node_index:
                        pos[node] = (mid_x + x_level_diff * (node_s - mid_node_index),
                                     y_diff * (num_level - level_s))
                    node_s = node_s + 1
            elif (num_node % 2) == 0:
                for node in node_list:
                    if node_s <= mid_node_index:
                        pos[node] = (mid_x - x_level_diff / 2 - x_level_diff * (mid_node_index - node_s),
                                     y_diff * (num_level - level_s))
                    elif node_s > mid_node_index:
                        pos[node] = (mid_x + x_level_diff / 2 + x_level_diff * (node_s - mid_node_index - 1),
                                     y_diff * (num_level - level_s))
                    node_s = node_s + 1
        level_s = level_s + 1
    return pos


def hierarchy_pos_horizontal(node_level_dict, x_diff, y_diff_max_node_set):
    num_level = len(node_level_dict)
    pos = {}
    level_s = 0
    num_node_level = [len(level_node) for level_node in node_level_dict.values()]
    max_num_node_level = max(num_node_level)
    max_y = max_num_node_level * y_diff_max_node_set  # no need to use "max_num_node_level-1"
    mid_y = max_y / 2
    for node_list in node_level_dict.values():
        num_node = len(node_list)
        if num_node == 0:
            print("Network ends at this level.")
        elif num_node == 1:
            pos[node_list[0]] = (x_diff * level_s, mid_y)
        elif num_node > 1:
            mid_node_index = np.ceil(num_node / 2)
            node_s = 1
            y_level_diff = max_y / (num_node - 1)
            if (num_node % 2) == 1:
                for node in node_list:
                    if node_s < mid_node_index:
                        pos[node] = (x_diff * level_s,
                                     mid_y - y_level_diff * (mid_node_index - node_s))
                    elif node_s >= mid_node_index:
                        pos[node] = (x_diff * level_s,
                                     mid_y + y_level_diff * (node_s - mid_node_index))
                    node_s = node_s + 1
            elif (num_node % 2) == 0:
                for node in node_list:
                    if node_s <= mid_node_index:
                        pos[node] = (x_diff * level_s,
                                     mid_y - y_level_diff / 2 - y_level_diff * (mid_node_index - node_s))
                    elif node_s > mid_node_index:
                        pos[node] = (x_diff * level_s,
                                     mid_y + y_level_diff / 2 + y_level_diff * (node_s - mid_node_index - 1))
                    node_s = node_s + 1

        level_s = level_s + 1
    return pos


def hierarchy_pos_horizontal_old(node_level_dict, x_diff, y_diff_max_node_set):
    num_level = len(node_level_dict)
    pos = {}
    level_s = 0
    num_node_level = [len(level_node) for level_node in node_level_dict.values()]
    max_num_node_level = max(num_node_level)
    max_y = max_num_node_level * y_diff_max_node_set  # no need to use "max_num_node_level-1"
    mid_y = max_y / 2
    for node_list in node_level_dict.values():
        num_node = len(node_list)
        if num_node != 0:
            mid_node_index = np.ceil(num_node / 2)
            if num_node == 1:
                pos[node_list[0]] = (x_diff * level_s, mid_y)
            if num_node > 1:
                node_s = 1
                y_level_diff = max_y / (num_node - 1)
                if (num_node % 2) == 1:
                    for node in node_list:
                        if node_s < mid_node_index:
                            pos[node] = (x_diff * level_s,
                                         mid_y - y_level_diff * (mid_node_index - node_s))
                        elif node_s >= mid_node_index:
                            pos[node] = (x_diff * level_s,
                                         mid_y + y_level_diff * (node_s - mid_node_index))
                        node_s = node_s + 1
                elif (num_node % 2) == 0:
                    for node in node_list:
                        if node_s <= mid_node_index:
                            pos[node] = (x_diff * level_s,
                                         mid_y - y_level_diff / 2 - y_level_diff * (mid_node_index - node_s))
                        elif node_s > mid_node_index:
                            pos[node] = (x_diff * level_s,
                                         mid_y + y_level_diff / 2 + y_level_diff * (node_s - mid_node_index - 1))
                        node_s = node_s + 1
        else:
            print("Network ends at this level.")
        level_s = level_s + 1
    return pos


def contribution_sn_number(G, risk, disease):
    # sn: shared neighbors
    shared_neighbors = list(nx.common_neighbors(G, risk, disease))
    sn_contribute = dict(zip(shared_neighbors, [1] * len(shared_neighbors)))
    df_sn_contribute = pd.DataFrame.from_dict(sn_contribute, orient='index', columns=['contribution'])
    # df_sn_contribute = df_sn_contribute.sort_values(by=['contribution'], ascending=False)
    return df_sn_contribute


def contribution_sn_weighted(G, ori_adj, var_names, risk, disease):
    # sn: shared neighbors
    shared_neighbors = list(nx.common_neighbors(G, risk, disease))
    risk_idx_in_original = np.where(var_names == risk)
    disease_idx_in_original = np.where(var_names == disease)

    sn_contribute = {}
    for sn in shared_neighbors:
        sn_idx_in_original = np.where(var_names == sn)
        risk_sn = ori_adj[risk_idx_in_original, sn_idx_in_original]
        disease_sn = ori_adj[disease_idx_in_original, sn_idx_in_original]
        sn_weight = (risk_sn + disease_sn) / 2
        sn_contribute[sn] = [item for subset in sn_weight for item in subset]

    df_sn_contribute = pd.DataFrame.from_dict(sn_contribute, orient='index', columns=['contribution'])
    df_sn_contribute = df_sn_contribute.sort_values(by=['contribution'], ascending=False)
    return df_sn_contribute


def contribution_sn_weighted_centrality(G, ori_adj, centrality, var_names, risk, disease):
    # sn: shared neighbors
    shared_neighbors = list(nx.common_neighbors(G, risk, disease))
    risk_idx_in_original = np.where(var_names == risk)
    disease_idx_in_original = np.where(var_names == disease)

    sn_contribute = {}
    for sn in shared_neighbors:
        sn_idx_in_original = np.where(var_names == sn)  # nei: neighbor
        key_cen_link_risk = (risk, sn)
        key_cen_link_disease = (disease, sn)
        if key_cen_link_risk not in centrality.keys():
            key_cen_link_risk = (sn, risk)
        if key_cen_link_disease not in centrality.keys():
            key_cen_link_disease = (sn, disease)

        edge_cen_risk = centrality[key_cen_link_risk]
        edge_cen_disease = centrality[key_cen_link_disease]

        risk_sn_cen = ori_adj[risk_idx_in_original, sn_idx_in_original] * edge_cen_risk
        disease_sn_cen = ori_adj[disease_idx_in_original, sn_idx_in_original] * edge_cen_disease

        sn_weight_cen = risk_sn_cen + disease_sn_cen
        sn_contribute[sn] = [item for subset in sn_weight_cen for item in subset]

    df_sn_contribute = pd.DataFrame.from_dict(sn_contribute, orient='index', columns=['contribution'])
    df_sn_contribute = df_sn_contribute.sort_values(by=['contribution'], ascending=False)
    return df_sn_contribute


def contribution_spr_neighbor_exp(G, next_dist_G, ori_adj, var_names, risk, disease):
    nei_spr_contribute = {}
    risk_idx_in_original = np.where(var_names == risk)
    disease_idx_in_original = np.where(var_names == disease)

    neighbors_risk = list(G.neighbors(risk))
    neighbors_disease = list(G.neighbors(disease))

    for nei_risk in neighbors_risk:
        nei_risk_idx = np.where(var_names == nei_risk)
        weight_risk_nei = ori_adj[risk_idx_in_original, nei_risk_idx]
        for nei_disease in neighbors_disease:
            if nx.has_path(next_dist_G, source=nei_risk, target=nei_disease):
                nei_disease_idx = np.where(var_names == nei_disease)
                weight_disease_nei = ori_adj[disease_idx_in_original, nei_disease_idx]
                sp_len = nx.shortest_path_length(next_dist_G, source=nei_risk, target=nei_disease, weight='weight')
                nei_spr_weight = (weight_risk_nei + weight_disease_nei) * np.exp(-sp_len)
                if nei_risk == nei_disease:
                    nei_spr_contribute[nei_risk] = [item for subset in nei_spr_weight for item in subset]
                else:
                    nei_spr_contribute[(nei_risk, nei_disease)] = [item for subset in nei_spr_weight for item in subset]
            else:
                continue

    df_nei_contribute = pd.DataFrame.from_dict(nei_spr_contribute, orient='index', columns=['contribution'])
    df_nei_contribute = df_nei_contribute.sort_values(by=['contribution'], ascending=False)
    return df_nei_contribute


def total_contribution_var_sn_number(G, risk_vars, disease_vars):
    ttl_contribution = pd.DataFrame(columns=['contribution'])
    for risk in risk_vars:
        for disease in disease_vars:
            contribution = contribution_sn_number(G, risk, disease)
            ttl_contribution = pd.concat([ttl_contribution, contribution])
    ttl_contribution = ttl_contribution.groupby(ttl_contribution.index).sum()
    ttl_contribution = ttl_contribution.sort_values(by=['contribution'], ascending=False)
    return ttl_contribution


def total_contribution_var_sn(G, ori_adj, var_names, risk_vars, disease_vars):
    ttl_contribution = pd.DataFrame(columns=['contribution'])
    for risk in risk_vars:
        if risk in G.nodes:
            for disease in disease_vars:
                if disease in G.nodes:
                    contribution = contribution_sn_weighted(G, ori_adj, var_names, risk, disease)
                    if contribution.empty:
                        continue
                    elif ttl_contribution.empty:
                        ttl_contribution = contribution.copy()
                    else:
                        ttl_contribution = pd.concat([ttl_contribution, contribution])
                else:
                    continue
        else:
            continue

    ttl_contribution = ttl_contribution.groupby(ttl_contribution.index).sum()
    ttl_contribution = ttl_contribution.sort_values(by=['contribution'], ascending=False)
    return ttl_contribution


def total_contribution_var_sn_centrality(G, ori_adj, centrality, var_names, risk_vars, disease_vars):
    ttl_contribution = pd.DataFrame(columns=['contribution'])
    for risk in risk_vars:
        for disease in disease_vars:
            # contribution = contribution_sn_weighted(G, ori_adj, var_names, risk, disease)
            contribution = contribution_sn_weighted_centrality(G, ori_adj, centrality, var_names, risk, disease)
            ttl_contribution = pd.concat([ttl_contribution, contribution])
    ttl_contribution = ttl_contribution.groupby(ttl_contribution.index).sum()
    ttl_contribution = ttl_contribution.sort_values(by=['contribution'], ascending=False)
    return ttl_contribution


def total_contribution_var_spr_exp(G, next_dist_G, ori_adj, var_names, risk_vars, disease_vars):
    ttl_contribution = pd.DataFrame(columns=['contribution'])
    for risk in risk_vars:
        for disease in disease_vars:
            contribution = contribution_spr_neighbor_exp(G, next_dist_G, ori_adj, var_names, risk, disease)
            ttl_contribution = pd.concat([ttl_contribution, contribution])
    ttl_contribution = ttl_contribution.groupby(ttl_contribution.index).sum()
    ttl_contribution = ttl_contribution.sort_values(by=['contribution'], ascending=False)
    return ttl_contribution


def count_times_var_top_contribution(G, ori_adj, var_names, risk_vars, disease_vars, top_n):
    ttl_contribution = pd.DataFrame(columns=['contribution'])
    for risk in risk_vars:
        if risk in G.nodes:
            for disease in disease_vars:
                if disease in G.nodes:
                    contribution = contribution_sn_weighted(G, ori_adj, var_names, risk, disease)
                    if contribution.empty:
                        continue
                    elif ttl_contribution.empty:
                        if len(contribution) <= top_n:
                            ttl_contribution = contribution.copy()
                        else:
                            contribution_top_n = contribution.iloc[:top_n]
                            ttl_contribution = contribution_top_n.copy()
                    else:
                        if len(contribution) <= top_n:
                            ttl_contribution = pd.concat([ttl_contribution, contribution])
                        else:
                            contribution_top_n = contribution.iloc[:top_n]
                            ttl_contribution = pd.concat([ttl_contribution, contribution_top_n])
                else:
                    continue
        else:
            continue

    times_var_contribution = pd.DataFrame(ttl_contribution.groupby(ttl_contribution.index).size(), columns=['times'])
    times_var_contribution.sort_values(by=['times'], ascending=False, inplace=True)
    return times_var_contribution


def count_times_var_top_contribution_cen(G, centrality, ori_adj, var_names, risk_vars, disease_vars, top_n):
    ttl_contribution = pd.DataFrame(columns=['contribution'])
    for risk in risk_vars:
        for disease in disease_vars:
            contribution = contribution_sn_weighted_centrality(G, ori_adj, centrality, var_names, risk, disease)
            if len(contribution) <= top_n:
                ttl_contribution = pd.concat([ttl_contribution, contribution])
            else:
                contribution_top_n = contribution.iloc[:top_n]
                ttl_contribution = pd.concat([ttl_contribution, contribution_top_n])

    times_var_contribution = pd.DataFrame(ttl_contribution.groupby(ttl_contribution.index).size(), columns=['times'])
    times_var_contribution.sort_values(by=['times'], ascending=False, inplace=True)
    return times_var_contribution


def count_times_var_top_contribution_spr(G, next_dist_G, ori_adj, var_names, risk_vars, disease_vars, top_n):
    ttl_contribution = pd.DataFrame(columns=['contribution'])
    for risk in risk_vars:
        for disease in disease_vars:
            contribution = contribution_spr_neighbor_exp(G, next_dist_G, ori_adj, var_names, risk, disease)
            if len(contribution) <= top_n:
                ttl_contribution = pd.concat([ttl_contribution, contribution])
            else:
                contribution_top_n = contribution.iloc[:top_n]
                ttl_contribution = pd.concat([ttl_contribution, contribution_top_n])

    times_var_contribution = pd.DataFrame(ttl_contribution.groupby(ttl_contribution.index).size(), columns=['times'])
    times_var_contribution.sort_values(by=['times'], ascending=False, inplace=True)
    return times_var_contribution


def new_attr_to_list(arr):
    new_attr = []
    n_node = arr.shape[0]
    for ii in range(n_node):
        for jj in range(n_node):
            if ii < jj:
                attr_value = arr[ii, jj]
                if np.isnan(attr_value):
                    continue
                else:
                    new_attr.append(attr_value)
    return new_attr


def bi_graph_create(adj_mat, var_group_1: list, var_group_2: list, var_names):
    """
    get adjacency matrix from the original one to create bipartite graph
    :param adj_mat: original adjacency matrix, full-connected graph.
    :param var_group_1: partite 1, variable list
    :param var_group_2: partite 2, variable list
    :param var_names:
    :return: adjacency matrix for the bipartite graph.
    """
    bipartite_all_links = set(product(*[var_group_1, var_group_2]))
    bipartite_net_adj = bipartite_adj_func(adj_mat, bipartite_all_links, var_names)
    df_bipartite_adj = pd.DataFrame(bipartite_net_adj, index=var_names, columns=var_names)
    bi_G = nx.from_pandas_adjacency(df_bipartite_adj)
    bi_G.remove_nodes_from(list(set(var_names) - set.union(set(var_group_1), set(var_group_2))))
    # weight = np.array([bi_G[u][v]['weight'] for u, v in bi_G.edges])
    print("The number of nodes in the network is", len(bi_G.nodes))
    return df_bipartite_adj, bi_G


def add_node_attribute(table, var_groups, attr_label):
    new_attr_val = [None] * len(table)
    table.insert(len(table.columns), attr_label, new_attr_val, True)
    for index, row in table.iterrows():
        for ii in range(len(var_groups)):
            if row['Id'] in var_groups[ii]:
                table.at[index, attr_label] = ii
    return table
