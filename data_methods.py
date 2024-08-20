import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KernelDensity
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from sklearn import linear_model
import scipy.stats as ss
from astropy.stats import bayesian_blocks
from scipy import stats
from itertools import combinations
import entropy_estimators as ee


def num_type(df_data, categorical_limit_num):
    var_type = []
    num_val = len(df_data)
    for col in list(df_data):
        # NOTE: integer in np.array is type "np.int64"; integer in list is type "int"
        col_value = df_data[col]
        col_uni, col_cnt = np.unique(col_value, return_counts=True)
        col_uni = col_uni[np.logical_not(np.isnan(col_uni))]  # remove nan from col_uni
        fraction_part = [math.modf(x)[0] for x in col_uni]
        if all(isinstance(x, np.int64) for x in col_uni) or not np.any(fraction_part):
            if len(col_uni) <= categorical_limit_num:
                var_type.append('c')  # categorical
            else:
                var_type.append('p')  # poisson
        else:
            var_type.append('g')  # gaussian

    return var_type


def impute_mixed_var(df_data, rv_type):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for rvs in list(df_data):
        if rv_type[rvs] == 'g':
            data_impute = imp_mean.fit_transform(pd.DataFrame(df_data[rvs]))
        elif rv_type[rvs] == 'p':
            data_impute = imp_med.fit_transform(pd.DataFrame(df_data[rvs]))
        else:
            data_impute = imp_freq.fit_transform(pd.DataFrame(df_data[rvs]))
        df_data.loc[:, rvs] = data_impute
    return df_data


def impute_mixed_var_new(df_data, rv_type):
    imp_knn = KNNImputer(n_neighbors=2)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for rvs in list(df_data):
        data_impute = imp_mean.fit_transform(pd.DataFrame(df_data[rvs]))
        # data_impute = imp_knn.fit_transform(pd.DataFrame(df_data[rvs]))
        # if rv_type[rvs] == 'g':
        #     # data_impute = imp_knn.fit_transform(pd.DataFrame(df_data[rvs]))
        #     data_impute = imp_mean.fit_transform(pd.DataFrame(df_data[rvs]))
        # elif rv_type[rvs] == 'p':
        #     data_impute = imp_med.fit_transform(pd.DataFrame(df_data[rvs]))
        # else:
        #     data_impute = imp_freq.fit_transform(pd.DataFrame(df_data[rvs]))
        df_data[rvs] = data_impute
    return df_data


# Function that computes the bins with "Bayesian Blocks"
def bin_maker(col):
    bins = bayesian_blocks(col)
    return np.searchsorted(bins, col)


def bin_maker_np(col, bin_method):
    bins_data = np.histogram_bin_edges(col, bins=bin_method)
    return np.searchsorted(bins_data, col)


def convert_to_categorical_df(df: pd.DataFrame, num_bin: int, method='equal_freq'):
    df_cat = df.copy()
    for col_name in df.columns.values:
        col_uni, col_cnt = np.unique(df_cat[col_name], return_counts=True)
        fraction_part = [math.modf(x)[0] for x in col_uni]
        if all(isinstance(x, int) for x in col_uni) or not np.any(fraction_part):
            if len(col_uni) > num_bin or max(col_uni) > num_bin:
                if method == 'equal_freq':
                    df_cat[col_name] = pd.qcut(df_cat[col_name], q=num_bin, labels=False, duplicates='drop')
                elif method == 'equal_bins':
                    df_cat[col_name] = pd.cut(df_cat[col_name], bins=num_bin, labels=False)
                else:
                    print('ValueError: unrecognized method=?')
            else:
                continue  # don't discretize for column that are int AND number of states/max(states) lower than num_bin
        else:
            if method == 'equal_freq':
                df_cat[col_name] = pd.qcut(df_cat[col_name], q=num_bin, labels=False, duplicates='drop')
            elif method == 'equal_bins':
                df_cat[col_name] = pd.cut(df_cat[col_name], bins=num_bin, labels=False)
            else:
                print('ValueError: unrecognized method=?')

    return df_cat


def pdf_multi_var(rv_mat, rv_type):
    """
    pdf estimation for mixed types of data.
    :param rv_mat: matrix in which each col stands for a random variable.
    :param rv_type: types of random variables
    :return: val_distribution -> pdf dictionary.
    """
    factor = 5
    rv_num = len(rv_type)
    val_distribution = {}
    for ii in range(rv_num):
        rv_data = rv_mat[:, ii][:, np.newaxis]
        x_plot = np.linspace(min(rv_data), max(rv_data), 500)
        if rv_type[ii] == 'g':
            bandwidth_range = list(np.linspace(min(rv_data) / factor, max(rv_data) / factor, num=5))
            grid_param = {'bandwidth': bandwidth_range, 'kernel': ['gaussian', 'epanechnikov']}
            kde_grid = sklearn.model_selection.GridSearchCV(KernelDensity(), grid_param)
            kde = kde_grid.fit(rv_data).best_estimator_
            pdf_val = list(np.exp(kde.score_samples(x_plot)))
        elif rv_type[ii] == 'p':
            start_bin = np.around(min(rv_data) / factor)
            end_bin = np.around(max(rv_data) / factor)
            bandwidth_range = list(np.around(np.linspace(start_bin, end_bin, num=5)))
            grid_param = {'bandwidth': bandwidth_range, 'kernel': ['gaussian', 'epanechnikov']}
            kde_grid = sklearn.model_selection.GridSearchCV(KernelDensity(), grid_param)
            kde = kde_grid.fit(rv_data).best_estimator_
            pdf_val = list(np.exp(kde.score_samples(x_plot)))
            # plt.subplot()
            # plt.plot(x_plot, np.exp(kde.score_samples(x_plot)), '-')
        else:
            unique_int_val = np.unique(rv_data.astype(int))
            ele_fre = {}
            for int_val in unique_int_val:
                ele_fre['ele_' + str(int_val)] = (list(rv_data.astype(int)).count(int_val)) / len(rv_data)
            pdf_val = ele_fre
        val_distribution[rv_type.index[ii]] = pdf_val
    return val_distribution


def pmf_multi_rvs(df_data):
    rvs_pmf = {}
    for rvs in list(df_data):
        unique, cnt = np.unique(df_data[rvs], return_counts=True)
        u_fre = cnt / len(df_data)
        rvs_pmf[rvs] = u_fre
    return rvs_pmf


def cal_mi(df_data):
    num_var = df_data.shape[1]
    mi = np.zeros([num_var, num_var])
    for rv_f in range(num_var):
        for rv_t in range(num_var):
            mi[rv_f, rv_t] = mutual_info_classif(np.transpose(np.matrix(df_data.iloc[:, rv_f])),
                                                 np.array(df_data.iloc[:, rv_t]),
                                                 discrete_features=True)

    return mi


def cal_shannon_ent(counts):
    shan_ent = []
    for rv_f in list(counts):
        shan_ent.append(stats.entropy(counts[rv_f]))

    return shan_ent


# Calculate MI using functions in sklearn.feature_selection
def cal_normalized_mi(df_data, counts):
    num_var = df_data.shape[1]
    norm_mi = np.zeros([num_var, num_var])
    for rv_f in range(num_var):
        ent_rv_f = stats.entropy(counts[df_data.columns[rv_f]])
        for rv_t in range(num_var):
            ent_rv_t = stats.entropy(counts[df_data.columns[rv_t]])
            ent_min = np.min([ent_rv_f, ent_rv_t])
            norm_mi[rv_f, rv_t] = mutual_info_classif(np.transpose(np.matrix(df_data.iloc[:, rv_f])),
                                                      np.array(df_data.iloc[:, rv_t]),
                                                      discrete_features=True) / ent_min
    return norm_mi


# Calculate MI using functions in sklearn.cluster.metrics
def cal_mi_skl_cluster(df_data):
    num_var = df_data.shape[1]
    abs_mi = np.zeros([num_var, num_var])
    norm_mi = np.zeros([num_var, num_var])
    for rv_true in range(num_var):
        label_true = list(df_data.iloc[:, rv_true])
        for rv_pred in range(num_var):
            if rv_pred > rv_true:
                label_pred = list(df_data.iloc[:, rv_pred])
                abs_mi[rv_true, rv_pred] = metrics.mutual_info_score(label_true, label_pred)
                norm_mi[rv_true, rv_pred] = metrics.normalized_mutual_info_score(label_true, label_pred, average_method='min')

                # norm_mi[rv_true, rv_pred] = metrics.adjusted_mutual_info_score(label_true, label_pred)  # Negative
    return norm_mi, abs_mi


# Calculate MI using functions in sklearn.cluster.metrics
def cal_mi_skl_cluster_all_pair(df_data):
    # Resulting in a symmetric MI matrix whose diagonal elements are entropy values.
    num_var = df_data.shape[1]
    abs_mi = np.zeros([num_var, num_var])
    norm_mi = np.zeros([num_var, num_var])
    for rv_true in range(num_var):
        label_true = list(df_data.iloc[:, rv_true])
        for rv_pred in range(num_var):
            label_pred = list(df_data.iloc[:, rv_pred])
            abs_mi[rv_true, rv_pred] = metrics.mutual_info_score(label_true, label_pred)
            norm_mi[rv_true, rv_pred] = metrics.normalized_mutual_info_score(label_true, label_pred,
                                                                             average_method='min')
    return norm_mi, abs_mi


# Calculate MI using functions in sklearn.cluster.metrics
def cal_mi_skl_cluster_with_nan(df_data: pd.DataFrame, categorical_vars_list: list):
    var_names = df_data.columns.to_list()
    num_var = df_data.shape[1]
    data_point_each_pair = []
    abs_mi = np.zeros([num_var, num_var])
    norm_mi = np.zeros([num_var, num_var])
    p_val_mat = np.zeros([num_var, num_var])
    for rv_true in range(num_var):
        rv_true_name = var_names[rv_true]
        rv_true_df = df_data.loc[:, [rv_true_name]]
        rv_true_df.dropna(axis=0, how='any', inplace=True)
        if rv_true_name not in categorical_vars_list:
            num_bin_sturge = int(np.ceil(np.log2(len(rv_true_df)) + 1))
            rv_true_df = convert_to_categorical_df(rv_true_df, num_bin=num_bin_sturge, method='equal_freq')

        # label_true = list(df_data.iloc[:, rv_true])
        for rv_pred in range(num_var):
            if rv_pred > rv_true:
                rv_pred_name = var_names[rv_pred]
                rv_pred_df = df_data.loc[:, [rv_pred_name]]
                rv_pred_df.dropna(axis=0, how='any', inplace=True)
                if rv_pred_name not in categorical_vars_list:
                    num_bin_sturge = int(np.ceil(np.log2(len(rv_pred_df)) + 1))
                    rv_pred_df = convert_to_categorical_df(rv_pred_df, num_bin=num_bin_sturge, method='equal_freq')
            # else:
            #     # rv_pred_df = pd.DataFrame()
            #     continue
                pair_discrete_df = pd.concat([rv_true_df, rv_pred_df], axis=1, join="inner").astype('int')
                data_point_each_pair.append(len(pair_discrete_df))
                rv_true_data = pair_discrete_df.iloc[:, 0]
                rv_pred_data = pair_discrete_df.iloc[:, 1]
                _, p_val_mat[rv_true, rv_pred], _, _ = ss.chi2_contingency(pd.crosstab(rv_true_data, rv_pred_data),
                                                                           lambda_='log-likelihood')
                label_true = list(rv_true_data)
                label_pred = list(rv_pred_data)

                abs_mi[rv_true, rv_pred] = metrics.mutual_info_score(label_true, label_pred)
                norm_mi[rv_true, rv_pred] = metrics.normalized_mutual_info_score(label_true, label_pred,
                                                                                 average_method='min')
    return norm_mi, abs_mi, p_val_mat, data_point_each_pair


def cal_mi_ee(df_data):
    # using the entropy_estimators in npeet package.
    # all info measures are in bit.
    num_var = df_data.shape[1]
    mi_mat = np.zeros([num_var, num_var])
    for rv_f in range(num_var):
        for rv_t in range(num_var):
            mi_mat[rv_f, rv_t] = ee.midd(list(df_data.iloc[:, rv_f]),
                                         list(df_data.iloc[:, rv_t]))
    return mi_mat


def cal_cc_pval(df_data, method: str):
    num_var = df_data.shape[1]
    cc = np.zeros([num_var, num_var])
    p_val = np.zeros([num_var, num_var])
    for rv_f in range(num_var):
        rv_f_array = df_data.iloc[:, rv_f].to_numpy()
        for rv_t in range(num_var):
            if rv_t > rv_f:
                rv_t_array = df_data.iloc[:, rv_t].to_numpy()
                if method == "Pearson":
                    corr_pval = stats.pearsonr(rv_f_array, rv_t_array)
                elif method == "Spearman":
                    corr_pval = stats.spearmanr(rv_f_array, rv_t_array)
                else:
                    corr_pval = []
                    print("Method not included.")
                cc[rv_f, rv_t] = corr_pval[0]
                p_val[rv_f, rv_t] = corr_pval[1]

    return cc, p_val


def cal_mi_comb(df_data, n_set):
    num_var = df_data.shape[1]
    comb = list(combinations(range(num_var), n_set))
    num_comb = len(comb)
    mi_comb = []
    mi_comb_mean = []
    mi_comb_max = []
    mi_comb_min = []
    for ii in range(num_comb):
        print(ii)
        mi_pairs = list(combinations(comb[ii], 2))
        mi_pairs_num = len(mi_pairs)
        mi_subset = []
        for jj in range(mi_pairs_num):
            mi_subset.append(mutual_info_classif(np.transpose(np.matrix(df_data.iloc[:, mi_pairs[jj][0]])),
                                                 np.array(df_data.iloc[:, mi_pairs[jj][1]]),
                                                 discrete_features=True))
            # mi_subset.append(ee.midd(list(df_data.iloc[:, mi_pairs[jj][0]]),
            #                          list(df_data.iloc[:, mi_pairs[jj][1]]))*np.log(2))
        mi_comb.append(mi_subset)
        mi_comb_mean.append(np.mean(mi_subset))
        mi_comb_max.append(np.max(mi_subset))
        mi_comb_min.append(np.min(mi_subset))

    dict_mi_comb = {'Combinations': comb, 'mi_comb': mi_comb,
                    'mean(mi)': mi_comb_mean, 'max(mi)': mi_comb_max,
                    'min(mi)': mi_comb_min}
    df_mi_comb = pd.DataFrame(dict_mi_comb)

    return df_mi_comb


def count_multi_rvs(df_data):
    rvs_count = {}
    for rvs in list(df_data):
        unique, cnt = np.unique(df_data[rvs], return_counts=True)
        # u_fre = cnt / len(df_data)
        rvs_count[rvs] = cnt
    return rvs_count


def cal_normalized_mi_comb(df_data, counts, n_set):
    num_var = df_data.shape[1]
    comb = list(combinations(range(num_var), n_set))
    num_comb = len(comb)
    normalized_mi_comb = []
    normalized_mi_comb_mean = []
    normalized_mi_comb_max = []
    normalized_mi_comb_min = []
    for ii in range(num_comb):
        print(ii)
        mi_pairs = list(combinations(comb[ii], 2))
        mi_pairs_num = len(mi_pairs)
        normalized_mi_subset = []
        for jj in range(mi_pairs_num):
            ent_min_jj = np.min([stats.entropy(counts[df_data.columns[mi_pairs[jj][0]]]),
                                 stats.entropy(counts[df_data.columns[mi_pairs[jj][1]]])])
            normalized_mi_subset.append(mutual_info_classif(np.transpose(np.matrix(df_data.iloc[:, mi_pairs[jj][0]])),
                                                            np.array(df_data.iloc[:, mi_pairs[jj][1]]),
                                                            discrete_features=True) / ent_min_jj)
        normalized_mi_comb.append(normalized_mi_subset)
        normalized_mi_comb_mean.append(np.mean(normalized_mi_subset))
        normalized_mi_comb_max.append(np.max(normalized_mi_subset))
        normalized_mi_comb_min.append(np.min(normalized_mi_subset))

    dict_nmi_comb = {'Combinations': comb, 'normMI_comb': normalized_mi_comb,
                     'mean(Norm_MI)': normalized_mi_comb_mean,
                     'max(Norm_MI)': normalized_mi_comb_max,
                     'min(Norm_MI)': normalized_mi_comb_min}
    df_normalized_mi_comb = pd.DataFrame(dict_nmi_comb)

    return df_normalized_mi_comb


def cal_entropy_comb(counts, n_set):
    num_var = len(counts)
    comb = list(combinations(range(num_var), n_set))
    num_comb = len(comb)
    ent_comb = []
    ent_comb_mean = []
    ent_comb_max = []
    ent_comb_min = []
    for ii in range(num_comb):
        print(ii)
        var_comb = comb[ii]
        ent_subset = [stats.entropy(counts[list(counts.keys())[x]]) for x in var_comb]
        ent_comb.append(ent_subset)
        ent_comb_mean.append(np.mean(ent_subset))
        ent_comb_max.append(np.max(ent_subset))
        ent_comb_min.append(np.min(ent_subset))

    dict_ent_comb = {'Combinations': comb,
                     'comb_ent': ent_comb,
                     'mean_ent': ent_comb_mean,
                     'max_ent': ent_comb_max,
                     'min_ent': ent_comb_min}
    df_ent_comb = pd.DataFrame(dict_ent_comb)

    return df_ent_comb


def rm_var_high_norm_mi(df_data, counts, threshold):
    num_var = df_data.shape[1]
    drop_var_index = []
    for rv_f in range(num_var):
        ent_rv_f = stats.entropy(counts[df_data.columns[rv_f]])
        for rv_t in range(num_var):
            if rv_t == rv_f:
                continue
            ent_rv_t = stats.entropy(counts[df_data.columns[rv_t]])
            ent_min = np.min([ent_rv_f, ent_rv_t])
            norm_mi = mutual_info_classif(np.transpose(np.matrix(df_data.iloc[:, rv_f])),
                                          np.array(df_data.iloc[:, rv_t]),
                                          discrete_features=True) / ent_min
            if norm_mi > threshold:
                drop_var_index.extend([rv_f, rv_t])

    return list(set(drop_var_index))


def cal_cond_ent_comb_mi(mi, n_set, mean_min):
    num_var = len(mi)
    comb = list(combinations(range(num_var), n_set))
    num_comb = len(comb)
    cond_ent_comb_pair_01 = []
    cond_ent_comb_pair_02 = []
    cond_ent_comb_pair_12 = []
    min_mean_cond_ent_comb_pair_01 = []
    min_mean_cond_ent_comb_pair_02 = []
    min_mean_cond_ent_comb_pair_12 = []

    for ii in range(num_comb):
        # print(ii)
        triplet = comb[ii]
        groups = list(combinations(range(n_set), 2))
        groups_num = len(groups)
        cond_ent_pair_01 = []
        cond_ent_pair_02 = []
        cond_ent_pair_12 = []
        for jj in range(groups_num):
            pair = groups[jj]
            # H(X|Y) = H(X) - I(X;Y)  X: pair[0]; Y: pair[1]
            cond_ent_1 = mi[triplet[pair[0]], triplet[pair[0]]] - mi[triplet[pair[0]], triplet[pair[1]]]
            # H(Y|X) = H(Y) - I(X;Y)  X: pair[0]; Y: pair[1]
            cond_ent_2 = mi[triplet[pair[1]], triplet[pair[1]]] - mi[triplet[pair[0]], triplet[pair[1]]]
            if jj == 0:
                cond_ent_pair_01.extend([cond_ent_1, cond_ent_2])
            elif jj == 1:
                cond_ent_pair_02.extend([cond_ent_1, cond_ent_2])
            elif jj == 2:
                cond_ent_pair_12.extend([cond_ent_1, cond_ent_2])

        cond_ent_comb_pair_01.append(cond_ent_pair_01)
        cond_ent_comb_pair_02.append(cond_ent_pair_02)
        cond_ent_comb_pair_12.append(cond_ent_pair_12)
        if mean_min == 'min':
            min_mean_cond_ent_comb_pair_01.append(np.min(cond_ent_pair_01))
            min_mean_cond_ent_comb_pair_02.append(np.min(cond_ent_pair_02))
            min_mean_cond_ent_comb_pair_12.append(np.min(cond_ent_pair_12))
        elif mean_min == 'mean':
            min_mean_cond_ent_comb_pair_01.append(np.mean(cond_ent_pair_01))
            min_mean_cond_ent_comb_pair_02.append(np.mean(cond_ent_pair_02))
            min_mean_cond_ent_comb_pair_12.append(np.mean(cond_ent_pair_12))

    dict_cond_ent_comb = {'Combinations': comb,
                          'cond_ent_pair_01': cond_ent_comb_pair_01,
                          'min_mean_cond_ent_pair_01': min_mean_cond_ent_comb_pair_01,
                          'cond_ent_pair_02': cond_ent_comb_pair_02,
                          'min_mean_cond_ent_pair_02': min_mean_cond_ent_comb_pair_02,
                          'cond_ent_pair_12': cond_ent_comb_pair_12,
                          'min_mean_cond_ent_pair_12': min_mean_cond_ent_comb_pair_12}
    df_cond_ent_comb = pd.DataFrame(dict_cond_ent_comb)

    return df_cond_ent_comb


def mi_bootstrap_p_value(mydata, mi_mat, boot_num, resample_num, norm):
    num_var = len(mydata.columns)
    p_value = np.zeros([num_var, num_var])
    for rv_f in range(num_var):
        print(rv_f)
        data_f = np.array(mydata.iloc[:, rv_f])
        for rv_t in range(num_var):
            if rv_f < rv_t:
                data_t = np.array(mydata.iloc[:, rv_t])
                boot_mi = []
                for _ in range(boot_num):
                    boot_sample_f = np.random.choice(data_f, size=resample_num, replace=True)
                    boot_sample_t = np.random.choice(data_t, size=resample_num, replace=True)
                    if norm:
                        uni_f, cnt_f = np.unique(boot_sample_f, return_counts=True)
                        ent_rv_f = stats.entropy(cnt_f)
                        uni_t, cnt_t = np.unique(boot_sample_t, return_counts=True)
                        ent_rv_t = stats.entropy(cnt_t)
                        ent_min = np.min([ent_rv_f, ent_rv_t])
                        mi_val = mutual_info_classif(np.transpose(np.matrix(boot_sample_f)),
                                                     boot_sample_t, discrete_features=True) / ent_min
                    else:
                        mi_val = mutual_info_classif(np.transpose(np.matrix(boot_sample_f)),
                                                     boot_sample_t, discrete_features=True)
                    boot_mi.extend(mi_val)
                boot_mi_np = np.array(boot_mi)
                mi_ft = mi_mat[rv_f][rv_t]
                num_extreme_val = np.count_nonzero(boot_mi_np >= mi_ft)
                p_value[rv_f, rv_t] = num_extreme_val / boot_num

            else:
                continue

    return p_value


# Using new function in sklearn.cluster.metrics to calculate normalized mutual info.
def mi_bootstrap_p_value_new(mydata, mi_mat, boot_num, resample_num, norm):
    num_var = len(mydata.columns)
    p_value = np.zeros([num_var, num_var])
    for rv_f in range(num_var):
        print(rv_f)
        data_f = np.array(mydata.iloc[:, rv_f])
        for rv_t in range(num_var):
            if rv_f < rv_t:
                data_t = np.array(mydata.iloc[:, rv_t])
                boot_mi = []
                for _ in range(boot_num):
                    boot_sample_f = np.random.choice(data_f, size=resample_num, replace=True)
                    boot_sample_t = np.random.choice(data_t, size=resample_num, replace=True)
                    if norm:
                        mi_val = metrics.normalized_mutual_info_score(boot_sample_f, boot_sample_t,
                                                                      average_method='min')
                    else:
                        mi_val = metrics.mutual_info_score(boot_sample_f, boot_sample_t)
                    boot_mi.append(mi_val)
                boot_mi_np = np.array(boot_mi)
                mi_ft = mi_mat[rv_f][rv_t]
                num_extreme_val = np.count_nonzero(boot_mi_np >= mi_ft)
                p_value[rv_f, rv_t] = num_extreme_val / boot_num

            else:
                continue
    return p_value


def df_yfs_data_age(df_yfs, sex):
    age_groups = np.unique(df_yfs['ika07'])
    for age in age_groups:
        print(age)
        df_yfs_age = df_yfs[df_yfs['ika07'] == age].drop(columns=['ika07'])
        # Replace ALL-NaN columns with zero.
        all_nan_columns = df_yfs_age.isna().all()
        df_yfs_age.update(df_yfs_age.loc[:, all_nan_columns].fillna(0))

        val_type = num_type(df_yfs_age, categorical_limit_num=5)
        var_type_dict = dict(zip(df_yfs_age.columns, val_type))
        impute_mixed_var(df_yfs_age, var_type_dict)  # warning from all nan column
        df_discrete_age = df_yfs_age.apply(bin_maker)  # warning from all zero column
        # data_count_age = count_multi_rvs(df_discrete_age)
        df_discrete_age.to_csv("df_discrete_data_yfs_" + sex + "_" + str(age) + ".csv")

        mi_norm_age, mi_abs_age = cal_mi_skl_cluster(df_discrete_age)
        # np.fill_diagonal(mi_norm_age, 0)
        # np.fill_diagonal(mi_abs_age, 0)
        pd.DataFrame(mi_norm_age).to_csv("mi_norm_" + sex + "_" + str(age) + ".csv")
        pd.DataFrame(mi_abs_age).to_csv("mi_absolute_" + sex + "_" + str(age) + ".csv")

        p_value_mi_age = mi_bootstrap_p_value_new(df_discrete_age, mi_abs_age, 100,
                                                  len(df_discrete_age), norm=False)
        pd.DataFrame(p_value_mi_age).to_csv("p_val_abs_mi_boot_100_" + sex + "_" + str(age) + ".csv")


def bio_var_filtering_mi(var_names, mi, mi_threshold):
    """
    Note: all phenotype variables should be collectively placed first before biomarker variables.
    :param var_names: all variables and their names
    :param mi: MI correlation matrix for these variables
    :param mi_threshold: the threshold set based on MI distribution
    :return: variables to be removed by the method.
    """
    first_biomarker_name = 'XXLVLDLP07'
    first_biomarker_idx = np.where(var_names == first_biomarker_name)[0][0]
    remove_var_list = []
    remove_col_idx = []  # given ii-s, index of columns removed by the method.
    for ii in range(first_biomarker_idx, len(var_names)):
        remove_col_idx_ii = []
        for jj in range(first_biomarker_idx, len(var_names)):
            if jj > ii and mi[ii, jj] > mi_threshold and jj not in remove_col_idx:
                var_ii_corr_sum = np.sum(mi[0:first_biomarker_idx, ii])
                var_jj_corr_sum = np.sum(mi[0:first_biomarker_idx, jj])
                if var_ii_corr_sum < var_jj_corr_sum and var_names[ii] not in set(remove_var_list):
                    remove_var_list.append(var_names[ii])
                    remove_col_idx_ii.clear()
                    break
                elif var_ii_corr_sum >= var_jj_corr_sum and var_names[jj] not in set(remove_var_list):
                    # remove_var_list.append(var_names[jj])
                    remove_col_idx_ii.append(jj)
                else:
                    print("Variable index error!")
            else:
                continue
        if remove_col_idx_ii:
            remove_var_list.extend(var_names[np.array(remove_col_idx_ii)].tolist())
            remove_col_idx.extend(remove_col_idx_ii)
    return remove_var_list


def cvd_var_filtering_mi(var_names, mi, mi_threshold):
    """
    Note: all phenotype variables should be collectively placed first before biomarker variables.
    :param var_names: all variables and their names
    :param mi: MI correlation matrix for these variables
    :param mi_threshold: the threshold set based on MI distribution
    :return: variables to be removed by the method.
    """
    first_cvd_name = 'dkv07'
    last_cvd_name = 'bbmax07'
    first_cvd_idx = np.where(var_names == first_cvd_name)[0][0]
    last_cvd_idx = np.where(var_names == last_cvd_name)[0][0]
    first_biomarker_name = 'XXLVLDLP07'
    first_biomarker_idx = np.where(var_names == first_biomarker_name)[0][0]
    remove_var_list = []
    remove_col_idx = []  # given ii-s, index of columns removed by the method.
    for ii in range(first_cvd_idx, last_cvd_idx + 1):
        remove_col_idx_ii = []
        for jj in range(first_cvd_idx, last_cvd_idx + 1):
            if jj > ii and mi[ii, jj] > mi_threshold and jj not in remove_col_idx:
                var_ii_corr_sum = np.sum(mi[first_biomarker_idx:, ii])
                var_jj_corr_sum = np.sum(mi[first_biomarker_idx:, jj])
                if var_ii_corr_sum < var_jj_corr_sum and var_names[ii] not in set(remove_var_list):
                    remove_var_list.append(var_names[ii])
                    remove_col_idx_ii.clear()
                    break
                elif var_ii_corr_sum >= var_jj_corr_sum and var_names[jj] not in set(remove_var_list):
                    # remove_var_list.append(var_names[jj])
                    remove_col_idx_ii.append(jj)
                else:
                    print("Variable index error!")
            else:
                continue
        if remove_col_idx_ii:
            remove_var_list.extend(var_names[np.array(remove_col_idx_ii)].tolist())
            remove_col_idx.extend(remove_col_idx_ii)
    return remove_var_list


def bio_var_filter_mi(var_names, mi, mi_threshold):
    """
    Note: all phenotype variables should be collectively placed first before biomarker variables.
    :param var_names: all variables and their names
    :param mi: MI correlation matrix for these variables
    :param mi_threshold: the threshold set based on MI distribution
    :return: variables to be removed by the method.
    """
    first_biomarker_name = 'XXLVLDLP07'
    first_biomarker_idx = np.where(var_names == first_biomarker_name)[0][0]
    remove_var_list = []
    for ii in range(first_biomarker_idx, len(var_names)):
        for jj in range(first_biomarker_idx, len(var_names)):
            if jj > ii and mi[ii, jj] > mi_threshold:
                var_ii_corr_sum = np.sum(mi[0:first_biomarker_idx, ii])
                var_jj_corr_sum = np.sum(mi[0:first_biomarker_idx, jj])
                if var_ii_corr_sum < var_jj_corr_sum and var_names[ii] not in set(remove_var_list):
                    remove_var_list.append(var_names[ii])
                elif var_ii_corr_sum >= var_jj_corr_sum and var_names[jj] not in set(remove_var_list):
                    remove_var_list.append(var_names[jj])
                else:
                    print("Variable index error!")
            else:
                continue
    return remove_var_list


def cvd_var_filter_mi(var_names, mi, mi_threshold):
    """
    Note: all phenotype variables should be collectively placed first before biomarker variables.
    :param var_names: all variables and their names
    :param mi: MI correlation matrix for these variables
    :param mi_threshold: the threshold set based on MI distribution
    :return: variables to be removed by the method.
    """
    first_cvd_name = 'dkv07'
    last_cvd_name = 'bbmax07'
    first_cvd_idx = np.where(var_names == first_cvd_name)[0][0]
    last_cvd_idx = np.where(var_names == last_cvd_name)[0][0]
    first_biomarker_name = 'XXLVLDLP07'
    first_biomarker_idx = np.where(var_names == first_biomarker_name)[0][0]
    remove_var_list = []
    for ii in range(first_cvd_idx, last_cvd_idx + 1):
        for jj in range(first_cvd_idx, last_cvd_idx + 1):
            if jj > ii and mi[ii, jj] > mi_threshold:
                var_ii_corr_sum = np.sum(mi[first_biomarker_idx:, ii])
                var_jj_corr_sum = np.sum(mi[first_biomarker_idx:, jj])
                if var_ii_corr_sum < var_jj_corr_sum and var_names[ii] not in set(remove_var_list):
                    remove_var_list.append(var_names[ii])
                elif var_ii_corr_sum >= var_jj_corr_sum and var_names[jj] not in set(remove_var_list):
                    remove_var_list.append(var_names[jj])
                else:
                    print("Variable index error!")
            else:
                continue
    return remove_var_list


def mi_p_val_chi2(df: pd.DataFrame):
    num_col = df.shape[1]
    p_val = np.zeros([num_col, num_col])
    for ii in range(num_col):
        for jj in range(num_col):
            if jj > ii:
                _, p_val[ii, jj], _, _ = ss.chi2_contingency(pd.crosstab(df.iloc[:, ii], df.iloc[:, jj]),
                                                             lambda_='log-likelihood')
            else:
                continue
    return p_val


def glm_coef_matrix(df: pd.DataFrame, glm_power=0.0, glm_alpha=1.0, glm_link='auto'):
    num_vars = df.shape[1]
    glm_coef_mat = np.zeros([num_vars, num_vars])
    glm_model = linear_model.TweedieRegressor(power=glm_power, alpha=glm_alpha, link=glm_link)
    for y_idx in range(num_vars):
        y = df.iloc[:, y_idx].values.tolist()
        for x_idx in range(num_vars):
            x = df.iloc[:, x_idx:x_idx + 1].values.tolist()
            glm_model.fit(x, y)
            glm_coef_mat[y_idx, x_idx] = glm_model.coef_[0]

    return glm_coef_mat
