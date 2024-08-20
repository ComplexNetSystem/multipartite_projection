import pandas as pd
import numpy as np
import data_methods as dm
from feature_engine.imputation import RandomSampleImputer
import pickle

# Prepare data like data frame after preprocessing, imputation, discretization, etc...
# Calculate normalized and absolute MI matrix and their p-value matrix.
# Save them in .csv files for use in the script for network analyses: "mi_net_filtering_rand_imp_loop.py"

# Number of imputations
imp_times = 20  # Times for random imputation.
num_existing_files = 0
SEX = False  # if SEX is empty, not to split data into two gender groups.
sex_groups = ["female", "male"]

MI_OR_CC = "CC"

# Replace each missing value with a random sample from known values.
rsi = RandomSampleImputer()

# Remove columns where 80% are missing.
remove_col_nan_percent = 0.8

# Variables to be removed, from new no imputation, removing NaN considering each single pair of vars.
with open("removed_vars_bio_cvd_no_imp_domain_0712.pkl", "rb") as f:
    remove_bio_var_list, remove_cvd_var_list = pickle.load(f)

# Names of data file to be saved.
yfs_data_name = "yfs_discrete_data_only_sym_0712_filter_qcut_sturges_rand_imp_"
# FOR MI
norm_mi_name = "mi_norm_yfs_0712_only_sym_filter_qcut_sturges_rand_imp_"
abs_mi_name = "mi_abs_yfs_0712_only_sym_filter_qcut_sturges_rand_imp_"
p_val_mi_name = "p_value_mi_yfs_chi2_0712_only_sym_filter_qcut_sturges_rand_imp_"
# FOR Correlation Coefficients
abs_cc_name = "cc_abs_yfs_0712_only_sym_filter_qcut_sturges_rand_imp_"
p_val_cc_name = "p_value_cc_yfs_chi2_0712_only_sym_filter_qcut_sturges_rand_imp_"

for ii in range(imp_times):
    print(ii)
    # 1. DEPRESSION SYMPTOMS and SUMMARY
    # In comment: data preparation, already saved in csv files used in following lines of codes.
    # discrete already Remove all-NaN rows, impute NaN values.
    depression_symptoms_df = pd.read_csv("symptoms2007.csv", index_col=0).dropna(axis=0, how='all')
    depression_symptoms_df = depression_symptoms_df.loc[:, depression_symptoms_df.isin([np.nan]).mean() < remove_col_nan_percent]
    depression_symptoms_df.replace([11, 12], 1, inplace=True)
    depression_symptoms_df.replace([21, 22], 2, inplace=True)
    depression_symptoms_df.replace([31, 32], 3, inplace=True)
    depression_symptoms_df = rsi.fit_transform(depression_symptoms_df)
    depression_symptoms_df = depression_symptoms_df.astype('int')
    # print(depression_symptoms_df.isnull().values.any())

    # discrete already, NO missing values
    depress_som_cog_df = pd.read_csv("somatic_cognitive_subgroup_score.csv", index_col=0).dropna(axis=0, how='all')
    # discrete already, but dynamic range is high, No missing values.
    depress_df = pd.read_csv("depression2007.csv", index_col=0).dropna(axis=0, how='all')
    depress_df.drop(['masennus_neliluokkainen'], axis=1, inplace=True)  # Remove duplicate depression variable

    # 2. PHENOTYPE VARIABLES
    phenotype_df = pd.read_csv("phenotypes2007.csv", index_col=0).dropna(axis=0, how='all')
    phenotype_df = phenotype_df.loc[:, phenotype_df.isin([np.nan]).mean() < remove_col_nan_percent]
    phenotype_df = rsi.fit_transform(phenotype_df)
    # print(phenotype_df.isnull().values.any())

    age_sex_df = phenotype_df.loc[:, ['ika07', 'SP']]

    phenotype_df.drop(columns=['ika07', 'SP'], inplace=True)
    categorical_pheno_df = phenotype_df.loc[:, ['idealCVH07', 'SESKOU07', 'smoke07']]
    categorical_pheno_df = categorical_pheno_df.astype('int')

    continuous_pheno_df = phenotype_df.drop(columns=['idealCVH07', 'SESKOU07', 'smoke07'])
    pheno_num_bin_sturge = int(np.ceil(np.log2(len(continuous_pheno_df)) + 1))
    discrete_continuous_pheno_df = dm.convert_to_categorical_df(continuous_pheno_df, num_bin=pheno_num_bin_sturge,
                                                                method='equal_freq')

    # 3. METABOLITES
    nmr_df = pd.read_csv("nmr2007.csv", index_col=0).dropna(axis=0, how='all')
    nmr_df = nmr_df.loc[:, nmr_df.isin([np.nan]).mean() < remove_col_nan_percent]
    nmr_df = rsi.fit_transform(nmr_df)
    nmr_num_bin_sturge = int(np.ceil(np.log2(len(nmr_df)) + 1))
    discrete_nmr_df = dm.convert_to_categorical_df(nmr_df, num_bin=nmr_num_bin_sturge, method='equal_freq')

    # 4. LIPIDS
    lipid_df_sample = pd.read_csv("lipidome2007.csv", index_col=0).dropna(axis=0, how='all')  # ild. "SAMPLE_NAME"
    lipid_df_sample = lipid_df_sample.loc[:, lipid_df_sample.isin([np.nan]).mean() < remove_col_nan_percent]
    lipid_df = lipid_df_sample.drop(columns=['SAMPLE_NAME'])  # drop "SAMPLE_NAME"
    lipid_df = rsi.fit_transform(lipid_df)
    lipid_num_bin_sturge = int(np.ceil(np.log2(len(lipid_df)) + 1))
    discrete_lipid_df = dm.convert_to_categorical_df(lipid_df, num_bin=lipid_num_bin_sturge, method='equal_freq')

    df_all_frames = [depression_symptoms_df, depress_som_cog_df, depress_df, age_sex_df,
                     categorical_pheno_df, discrete_continuous_pheno_df,
                     discrete_nmr_df, discrete_lipid_df]
    df_discrete_all_yfs = pd.concat(df_all_frames, axis=1, join="inner")

    # COMBINATION
    # Only symptoms
    df_frames = [depression_symptoms_df, age_sex_df,
                 categorical_pheno_df, discrete_continuous_pheno_df,
                 discrete_nmr_df, discrete_lipid_df]

    # Only summary
    # df_frames = [depress_df, age_sex_df, categorical_pheno_df, discrete_continuous_pheno_df,
    #              discrete_nmr_df, discrete_lipid_df]

    df_discrete_yfs = pd.concat(df_frames, axis=1, join="inner")
    print(np.array_equal(df_discrete_yfs, df_discrete_yfs.astype(int)))  # Check if all is integer.

    # change the order of "idealCVH07"
    ideal_CVH_data = df_discrete_yfs["idealCVH07"]  # df_discrete_yfs.loc[:,["idealCVH07"]]
    df_discrete_yfs.drop(columns=['idealCVH07'], inplace=True)
    df_discrete_yfs.insert(int(np.where(df_discrete_yfs.columns == 'imtka07')[0][0]),
                           "idealCVH07", ideal_CVH_data, allow_duplicates=False)

    df_discrete_yfs = df_discrete_yfs.loc[:, df_discrete_yfs.any()]  # Remove all-zero columns.
    print(df_discrete_yfs.shape)
    # var_names = df_discrete_yfs.columns.values

    # MI and p-value calculation
    df_discrete_yfs.drop(columns=remove_bio_var_list + remove_cvd_var_list, inplace=True)

    df_discrete_yfs_to_calculate = df_discrete_yfs.copy()
    file_suffix = ".csv"

    if SEX:
        for sex_idx in sex_groups:
            if sex_idx == 'female':
                df_discrete_yfs_sex_subset = df_discrete_yfs[df_discrete_yfs['SP'] == 1]
                file_suffix = "_female.csv"
            elif sex_idx == 'male':
                df_discrete_yfs_sex_subset = df_discrete_yfs[df_discrete_yfs['SP'] == 2]
                file_suffix = "_male.csv"
            else:
                print("Sex code error.")
                df_discrete_yfs_sex_subset = pd.DataFrame()

            df_discrete_yfs_to_calculate_sex = df_discrete_yfs_sex_subset.copy()
            df_discrete_yfs_to_calculate_sex.drop(columns=['SP'], inplace=True)
            print(df_discrete_yfs_to_calculate_sex.shape)

            df_discrete_yfs_to_calculate_sex.to_csv(yfs_data_name + str(num_existing_files + ii) + file_suffix)

            mi_norm_filter, mi_abs_filter = dm.cal_mi_skl_cluster(df_discrete_yfs_to_calculate_sex)
            pd.DataFrame(mi_norm_filter).to_csv(norm_mi_name + str(num_existing_files + ii) + file_suffix)
            pd.DataFrame(mi_abs_filter).to_csv(abs_mi_name + str(num_existing_files + ii) + file_suffix)

            p_value_mi = dm.mi_p_val_chi2(df_discrete_yfs_to_calculate_sex)
            pd.DataFrame(p_value_mi).to_csv(p_val_mi_name + str(num_existing_files + ii) + file_suffix)

    else:
        if MI_OR_CC == "MI":
            print(df_discrete_yfs_to_calculate.shape)
            df_discrete_yfs_to_calculate.to_csv(yfs_data_name + str(num_existing_files + ii) + file_suffix)

            mi_norm_filter, mi_abs_filter = dm.cal_mi_skl_cluster(df_discrete_yfs_to_calculate)
            pd.DataFrame(mi_norm_filter).to_csv(norm_mi_name + str(num_existing_files + ii) + file_suffix)
            pd.DataFrame(mi_abs_filter).to_csv(abs_mi_name + str(num_existing_files + ii) + file_suffix)

            p_value_mi = dm.mi_p_val_chi2(df_discrete_yfs_to_calculate)
            pd.DataFrame(p_value_mi).to_csv(p_val_mi_name + str(num_existing_files + ii) + file_suffix)
        elif MI_OR_CC == "CC":
            print(df_discrete_yfs_to_calculate.shape)
            df_discrete_yfs_to_calculate.to_csv(yfs_data_name + str(num_existing_files + ii) + file_suffix)

            corr_abs_filter, p_corr_filter = dm.cal_cc_pval(df_discrete_yfs_to_calculate, method="Pearson")
            pd.DataFrame(corr_abs_filter).to_csv(abs_cc_name + str(num_existing_files + ii) + ".csv")
            pd.DataFrame(p_corr_filter).to_csv(p_val_cc_name + str(num_existing_files + ii) + ".csv")
        else:
            print("Please specify the correlation TYPE!")

