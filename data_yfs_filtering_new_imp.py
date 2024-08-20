import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_methods as dm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle

# The best parameters were determined in "rf_model_gridsearch.py"
rf_regressor_estimator_nmr = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=0.002,
                                                   min_samples_split=2, min_samples_leaf=2)
rf_regressor_estimator_lipid = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=0.008,
                                                     min_samples_split=2, min_samples_leaf=4)
rf_regressor_estimator_cvd = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=0.002,
                                                   min_samples_split=2, min_samples_leaf=2)
rf_classifier_estimator = RandomForestClassifier(n_estimators=50, max_depth=4, max_features=0.002,
                                                 min_samples_split=2, min_samples_leaf=2)
rf_imputer_4_nmr = IterativeImputer(estimator=rf_regressor_estimator_nmr, max_iter=25)
rf_imputer_4_lipid = IterativeImputer(estimator=rf_regressor_estimator_lipid, max_iter=25)
rf_imputer_4_cvd = IterativeImputer(estimator=rf_regressor_estimator_cvd, max_iter=25)
rf_imputer_4_categorical = IterativeImputer(estimator=rf_classifier_estimator, max_iter=25)

imputer_continuous = IterativeImputer(estimator=LinearRegression())
imputer_categorical = IterativeImputer(estimator=LogisticRegression())
# imputer_continuous_br = IterativeImputer(estimator=BayesianRidge(), max_iter=25)  # Optimization, and Regularization
# imputer_categorical = IterativeImputer(estimator=KNeighborsClassifier(), max_iter=25)  # No convergence problem: KN

# Variables to be removed, from new no imputation, removing NaN considering each single pair of vars.
with open("removed_vars_bio_cvd_no_imp_domain_0712.pkl", "rb") as f:
    remove_bio_var_list, remove_cvd_var_list = pickle.load(f)

# In comment: data preparation, already saved in csv files used in following lines of codes.
# Remove all-NaN rows, impute NaN values.
# discrete already
depression_symptoms_df = pd.read_csv("symptoms2007.csv", index_col=0).dropna(axis=0, how='all')
depression_symptoms_df.replace([11, 12], 1, inplace=True)
depression_symptoms_df.replace([21, 22], 2, inplace=True)
depression_symptoms_df.replace([31, 32], 3, inplace=True)
imputed_depression_symptoms = imputer_categorical.fit_transform(depression_symptoms_df)
imputed_depression_symptoms = imputed_depression_symptoms.astype('int')
imputed_depression_symptoms_df = pd.DataFrame(imputed_depression_symptoms, index=depression_symptoms_df.index,
                                              columns=depression_symptoms_df.columns)

depress_som_cog_df = pd.read_csv("somatic_cognitive_subgroup_score.csv", index_col=0)  # discrete already

# discrete already, but dynamic range is high
depress_df = pd.read_csv("depression2007.csv", index_col=0).dropna(axis=0, how='all')
depress_df.drop(['masennus_neliluokkainen'], axis=1, inplace=True)  # Remove duplicate depression variable

# depress_sym_summ_df = pd.concat([depression_symptoms_df, depress_df], axis=1, join='inner').astype('int')
# print(depress_sym_summ_df.isnull().values.any())  # # after combining, No missing values any more.

phenotype_df = pd.read_csv("phenotypes2007.csv", index_col=0)
age_sex_df = phenotype_df.loc[:, ['ika07', 'SP']]
phenotype_df.drop(columns=['ika07', 'SP', 'volscore07'], inplace=True)
phenotype_df.dropna(axis=0, how='all', inplace=True)
# Remove patients whose CVD and BMI variables are all missing, added by Jie on April 16.
phenotype_df_bmi_cvd_no_all_nan = phenotype_df.iloc[:, :-3].dropna(axis=0, how='all')
phenotype_df_other_risk_no_all_nan = phenotype_df.iloc[:, -3:].dropna(axis=0, how='all')
phenotype_df_new = pd.concat([phenotype_df_bmi_cvd_no_all_nan, phenotype_df_other_risk_no_all_nan], axis=1,
                             join="inner")

categorical_pheno_df = phenotype_df_new.loc[:, ['idealCVH07', 'SESKOU07', 'smoke07']]
continuous_pheno_df = phenotype_df_new.drop(columns=['idealCVH07', 'SESKOU07', 'smoke07'])
# For categorical phenotypes
imputed_categorical_pheno = imputer_categorical.fit_transform(categorical_pheno_df)
imputed_categorical_pheno = imputed_categorical_pheno.astype('int')
imputed_categorical_pheno_df = pd.DataFrame(imputed_categorical_pheno, index=categorical_pheno_df.index,
                                            columns=categorical_pheno_df.columns)
# For continuous phenotypes
imputed_continuous_pheno = imputer_continuous.fit_transform(continuous_pheno_df)
imputed_continuous_pheno_df = pd.DataFrame(imputed_continuous_pheno, index=continuous_pheno_df.index,
                                           columns=continuous_pheno_df.columns)
pheno_num_bin_sturge = int(np.ceil(np.log2(len(imputed_continuous_pheno_df)) + 1))
imputed_discrete_continuous_pheno_df = dm.convert_to_categorical_df(imputed_continuous_pheno_df,
                                                                    num_bin=pheno_num_bin_sturge, method='equal_freq')

nmr_df = pd.read_csv("nmr2007.csv", index_col=0).dropna(axis=0, how='all')
imputed_nmr = imputer_continuous.fit_transform(nmr_df)
imputed_nmr_df = pd.DataFrame(imputed_nmr, index=nmr_df.index, columns=nmr_df.columns)
nmr_num_bin_sturge = int(np.ceil(np.log2(len(imputed_nmr_df)) + 1))
imputed_discrete_nmr_df = dm.convert_to_categorical_df(imputed_nmr_df, num_bin=nmr_num_bin_sturge, method='equal_freq')

lipid_df_sample = pd.read_csv("lipidome2007.csv", index_col=0)  # ild. "SAMPLE_NAME"
lipid_df = lipid_df_sample.drop(columns=['SAMPLE_NAME']).dropna(axis=0, how='all')  # drop "SAMPLE_NAME"
imputed_lipid = imputer_continuous.fit_transform(lipid_df)
imputed_lipid_df = pd.DataFrame(imputed_lipid, index=lipid_df.index, columns=lipid_df.columns)
lipid_num_bin_sturge = int(np.ceil(np.log2(len(imputed_lipid_df)) + 1))
imputed_discrete_lipid_df = dm.convert_to_categorical_df(imputed_lipid_df, num_bin=lipid_num_bin_sturge,
                                                         method='equal_freq')

# Only symptoms
df_frames = [imputed_depression_symptoms_df, age_sex_df,
             imputed_categorical_pheno_df, imputed_discrete_continuous_pheno_df,
             imputed_discrete_nmr_df, imputed_discrete_lipid_df]
df_discrete_yfs = pd.concat(df_frames, axis=1, join="inner")

# change the order of "idealCVH07"
ideal_CVH_data = df_discrete_yfs["idealCVH07"]  # df_discrete_yfs.loc[:,["idealCVH07"]]
df_discrete_yfs.drop(columns=['idealCVH07'], inplace=True)
df_discrete_yfs.insert(int(np.where(df_discrete_yfs.columns == 'imtka07')[0][0]),
                       "idealCVH07", ideal_CVH_data, allow_duplicates=False)

df_discrete_yfs = df_discrete_yfs.loc[:, df_discrete_yfs.any()]  # Remove all-zero columns.
var_names = df_discrete_yfs.columns.values
print(df_discrete_yfs.isnull().values.any())
print(df_discrete_yfs.shape)

# mi_norm, mi_abs = dm.cal_mi_skl_cluster(df_discrete_yfs)
# mi_threshold = 1.25
# remove_bio_var_list = dm.bio_var_filter_mi(var_names, mi_abs, mi_threshold)
# remove_cvd_var_list = dm.cvd_var_filter_mi(var_names, mi_abs, mi_threshold)

# MI and p-value calculation
df_discrete_yfs.drop(columns=remove_bio_var_list + remove_cvd_var_list, inplace=True)
print(df_discrete_yfs.shape)
df_discrete_yfs.to_csv("yfs_discrete_data_only_sym_0712_filter_qcut_sturges_impute_linear_50.csv")
mi_norm_filter, mi_abs_filter = dm.cal_mi_skl_cluster(df_discrete_yfs)
# np.fill_diagonal(mi_norm_filter, 0)
# np.fill_diagonal(mi_abs_filter, 0)
pd.DataFrame(mi_norm_filter).to_csv("mi_norm_yfs_0712_only_sym_filter_qcut_sturges_impute_linear_50.csv")
pd.DataFrame(mi_abs_filter).to_csv("mi_abs_yfs_0712_only_sym_filter_qcut_sturges_impute_linear_50.csv")

# p_value_mi = dm.mi_bootstrap_p_value_new(df_discrete_yfs, mi_abs_filter, 100, len(df_discrete_yfs), norm=False)
p_value_mi = dm.mi_p_val_chi2(df_discrete_yfs)
pd.DataFrame(p_value_mi).to_csv("p_value_mi_yfs_chi2_0712_only_sym_filter_qcut_sturges_impute_linear_50.csv")

"""
# Distribution of MI and normalized MI.
mi_abs_val_arr = mi_abs[np.triu_indices(len(mi_abs), 1)]
mi_norm_val_arr = mi_norm[np.triu_indices(len(mi_norm), 1)]
# plt.hist(mi_abs_val_arr, bins=50, density=True, alpha=0.6)
plt.hist(mi_abs_val_arr, bins=50, density=True, alpha=0.6)
plt.axvline(x=1.25, ls='--', c='k', lw=2)
plt.xlabel('MI', fontsize=20)
plt.ylabel('Probability density', fontsize=20)
# plt.xlim([-0.05, 1])
plt.xticks(np.arange(0, 2.6, step=0.5), fontsize=16)
# plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.yscale('log')
# plt.legend(['MI', 'Normalized MI'], fontsize=14)
plt.tight_layout()
plt.savefig("hist_density_abs_MI_only_summ_0328_qcut_sturges_impute_rf_50.pdf")
"""
