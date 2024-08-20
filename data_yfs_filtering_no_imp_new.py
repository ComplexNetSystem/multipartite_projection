import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_methods as dm
import pickle

# Without imputation, just remove the participants with missing values considering each pair when calculating MI.

remove_col_nan_percent = 0.8

# Variables to be removed, from new no imputation, removing NaN considering each single pair of vars.
with open("removed_vars_bio_cvd_no_imp_domain_0712.pkl", "rb") as f:
    remove_bio_var_list, remove_cvd_var_list = pickle.load(f)

# In comment: data preparation, already saved in csv files used in following lines of codes.
# Remove all-NaN rows, impute NaN values.
# discrete already
depression_symptoms_df = pd.read_csv("symptoms2007.csv", index_col=0).dropna(axis=0, how='all')
depression_symptoms_df = depression_symptoms_df.loc[:, depression_symptoms_df.isin([np.nan]).mean() < remove_col_nan_percent]
depression_symptoms_df.replace([11, 12], 1, inplace=True)
depression_symptoms_df.replace([21, 22], 2, inplace=True)
depression_symptoms_df.replace([31, 32], 3, inplace=True)
depress_sym_vars = depression_symptoms_df.columns.to_list()


# discrete already, but dynamic range is high
depress_som_cog_df = pd.read_csv("somatic_cognitive_subgroup_score.csv", index_col=0).dropna(axis=0, how='all')
depress_summ_df = pd.read_csv("depression2007.csv", index_col=0).dropna(axis=0, how='all')
depress_summ_df.drop(['masennus_neliluokkainen'], axis=1, inplace=True)  # Remove duplicate depression variable
depress_summ_vars = depress_som_cog_df.columns.to_list() + depress_summ_df.columns.to_list()


phenotype_df = pd.read_csv("phenotypes2007.csv", index_col=0).dropna(axis=0, how='all')
phenotype_df = phenotype_df.loc[:, phenotype_df.isin([np.nan]).mean() < remove_col_nan_percent]
pheno_categorical_vars = ['ika07', 'SP', 'SESKOU07', 'smoke07', 'idealCVH07']
pheno_continuous_vars = [var for var in phenotype_df.columns.to_list() if var not in pheno_categorical_vars]

# pheno_num_bin_sturge = int(np.ceil(np.log2(len(continuous_pheno_df)) + 1))
# discrete_continuous_pheno_df = dm.convert_to_categorical_df(continuous_pheno_df, num_bin=pheno_num_bin_sturge,
#                                                             method='equal_freq')


nmr_df = pd.read_csv("nmr2007.csv", index_col=0).dropna(axis=0, how='all')
nmr_df = nmr_df.loc[:, nmr_df.isin([np.nan]).mean() < remove_col_nan_percent]
nmr_vars = nmr_df.columns.to_list()
# nmr_df.dropna(axis=0, how='any', inplace=True)
# nmr_num_bin_sturge = int(np.ceil(np.log2(len(nmr_df)) + 1))
# discrete_nmr_df = dm.convert_to_categorical_df(nmr_df, num_bin=nmr_num_bin_sturge, method='equal_freq')


lipid_df_sample = pd.read_csv("lipidome2007.csv", index_col=0).dropna(axis=0, how='all')  # ild. "SAMPLE_NAME"
lipid_df_sample = lipid_df_sample.loc[:, lipid_df_sample.isin([np.nan]).mean() < remove_col_nan_percent]
lipid_df = lipid_df_sample.drop(columns=['SAMPLE_NAME'])  # drop "SAMPLE_NAME"
lipid_vars = lipid_df.columns.to_list()
# lipid_df.dropna(axis=0, how='any', inplace=True)
# lipid_num_bin_sturge = int(np.ceil(np.log2(len(lipid_df)) + 1))
# discrete_lipid_df = dm.convert_to_categorical_df(lipid_df, num_bin=lipid_num_bin_sturge, method='equal_freq')


categorical_vars = depress_sym_vars + depress_summ_vars + pheno_categorical_vars
continuous_vars = pheno_continuous_vars + nmr_vars + lipid_vars

# Only symptoms
df_frames = [depression_symptoms_df, phenotype_df, nmr_df, lipid_df]

# Only summary
# df_frames = [depress_summ_df, phenotype_df, nmr_df, lipid_df]

df_yfs_combine = pd.concat(df_frames, axis=1, join="outer")
df_yfs_combine = df_yfs_combine.loc[:, df_yfs_combine.any()]  # Remove all-zero columns.
print(df_yfs_combine.shape)
var_names = df_yfs_combine.columns.values


# mi_norm, mi_abs, p_value_mi, data_point_pair = dm.cal_mi_skl_cluster_with_nan(df_yfs_combine, categorical_vars)
# np.fill_diagonal(mi_norm, 0)
# np.fill_diagonal(mi_abs, 0)


# mi_threshold = 1.25
# remove_bio_var_list = dm.bio_var_filter_mi(var_names, mi_abs, mi_threshold)
# remove_cvd_var_list = dm.cvd_var_filter_mi(var_names, mi_abs, mi_threshold)


# with open("removed_vars_bio_cvd_no_imp_new.pkl", "wb") as f:
#     pickle.dump([remove_bio_var_list, remove_cvd_var_list], f)

# with open("removed_vars_bio_cvd_no_imp_new.pkl", "rb") as f:
#     remove_bio_var_list, remove_cvd_var_list = pickle.load(f)


# MI and p-value calculation
df_yfs_combine.drop(columns=remove_bio_var_list + remove_cvd_var_list, inplace=True)
print(df_yfs_combine.shape)
df_yfs_combine.to_csv("yfs_data_only_sym_0712_filter_qcut_sturges_no_imp_new.csv")

mi_norm_filter, mi_abs_filter, p_value_mi_filter, data_point_pair_filter = dm.cal_mi_skl_cluster_with_nan(df_yfs_combine, categorical_vars)
pd.DataFrame(mi_norm_filter).to_csv("mi_norm_yfs_0712_only_sym_filter_qcut_sturges_no_imp_new.csv")
pd.DataFrame(mi_abs_filter).to_csv("mi_abs_yfs_0712_only_sym_filter_qcut_sturges_no_imp_new.csv")
pd.DataFrame(p_value_mi_filter).to_csv("p_value_mi_yfs_chi2_0712_only_sym_filter_qcut_sturges_no_imp_new.csv")


"""
# Distribution of MI and normalized MI.
mi_abs_val_arr = mi_abs[np.triu_indices(len(mi_abs), 1)]
mi_norm_val_arr = mi_norm[np.triu_indices(len(mi_norm), 1)]
# plt.hist(mi_abs_val_arr, bins=50, density=True, alpha=0.6)
plt.hist(mi_norm_val_arr, bins=50, density=True, alpha=0.6)
plt.axvline(x=0.5, ls='--', c='k', lw=2)
plt.xlabel('Normalized MI', fontsize=20)
plt.ylabel('Probability density', fontsize=20)
# plt.xlim([-0.05, 1])
plt.xticks(np.arange(0, 1.2, step=0.25), fontsize=16)
# plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.yscale('log')
# plt.legend(['MI', 'Normalized MI'], fontsize=14)
plt.tight_layout()
plt.savefig("hist_density_norm_MI_only_sym_0424_qcut_sturges_no_imp_new.pdf")
"""
