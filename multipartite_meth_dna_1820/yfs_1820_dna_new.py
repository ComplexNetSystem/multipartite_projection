import pandas as pd
import numpy as np
import data_methods as dm
import seaborn as sns
import matplotlib.pyplot as plt

# This is the main script to prepare data such as discretization, MI calculation, ..., for projection
# For Pashu's project. YFS 2018-20 wave is processed.
# The new version has excluded chrX and chrY.
# On Sep. 18, 2025, by Jie.


SEX = False
sex_groups = ["female", "male"]

# Read .csv data files
meth_df = pd.read_csv("YFS_2018_2020_updated/meth_2025.csv", index_col=0).transpose()
clinical_df = pd.read_csv("YFS_2018_2020_updated/clinical_2025.csv", index_col=0)
clinical_df.drop(columns=['LDL_cholesterol', 'Diastolic_BP', 'Waist_size', 'Obesity'], inplace=True)

print(f'Any missing values? {meth_df.isnull().values.any()}')
print(f'Any missing values? {clinical_df.isnull().values.any()}')

len_meth, num_vars_meth = meth_df.shape
len_clinical, num_vars_clinical = clinical_df.shape

meth_vars = meth_df.columns.to_list()
clinical_vars = clinical_df.columns.to_list()

var_types_meth = dm.num_type(meth_df, categorical_limit_num=8)
var_types_clinical = dm.num_type(clinical_df, categorical_limit_num=8)

# Clinical data.
# Find continuous variables in the dataframe.
continuous_vars_clinical_idx = np.where(np.array(var_types_clinical) == 'g')[0]
continuous_vars_clinical = np.array(clinical_vars)[continuous_vars_clinical_idx]
continuous_clinical_df = clinical_df.iloc[:, list(continuous_vars_clinical_idx)]
# Discretize continuous variables using qcut with Sturge's rule
num_bins = int(np.ceil(np.log2(len_clinical) + 1))
discrete_conti_clinical_df = dm.convert_to_categorical_df(continuous_clinical_df, num_bin=num_bins, method='equal_freq')
# Replace continuous columns with discretized columns.
discrete_clinical_df = clinical_df.copy()
discrete_clinical_df.loc[:, continuous_vars_clinical] = discrete_conti_clinical_df
discrete_clinical_df = discrete_clinical_df.astype('int')


"""
# Check correlation between clinical variables and remove redundancy.
mi_norm_clinical, mi_abs_clinical = dm.cal_mi_skl_cluster(discrete_clinical_df)
hm_df = pd.DataFrame(mi_abs_clinical, columns=clinical_vars, index=clinical_vars)
plt.figure(figsize=(12, 8))
ax = sns.heatmap(hm_df, fmt=".2f", cmap='coolwarm', cbar=True)
ax.set_aspect('equal')
plt.tight_layout()
# plt.savefig("YFS_2018_2020/heatmap_abs_MI_correlation_clinical_vars.pdf")
"""


# Methylation data and discrete
num_bins = int(np.ceil(np.log2(len_meth) + 1))
discrete_meth_df = dm.convert_to_categorical_df(meth_df, num_bin=num_bins, method='equal_freq')


# Merge the two dataframes into one, to calculate MI between all pairs
# discrete_meth_df.index = discrete_meth_df.index.astype(int)    # patient ID of meth data was str, convert it to int.
clinical_meth_df_list = [discrete_clinical_df, discrete_meth_df]
df_discrete_yfs = pd.concat(clinical_meth_df_list, axis=1, join="inner")
print(f'After discretization: {df_discrete_yfs.shape}')

df_discrete_yfs_to_calculate = df_discrete_yfs.copy()

# Names of data file to be saved.
file_suffix = ".csv"
yfs_data_name = "YFS_2018_2020_updated/yfs_discrete_18_20_0919_qcut_sturges"
norm_mi_name = "YFS_2018_2020_updated/mi_norm_yfs_18_20_0919_qcut_sturges"
abs_mi_name = "YFS_2018_2020_updated/mi_abs_yfs_18_20_0919_qcut_sturges"
p_val_mi_name = "YFS_2018_2020_updated/mi_chi2_p_val_yfs_18_20_0919_qcut_sturges"

if SEX:
    for sex_idx in sex_groups:
        if sex_idx == 'female':
            df_discrete_yfs_sex_subset = df_discrete_yfs[df_discrete_yfs['Sex'] == 1]
            file_suffix = "_female.csv"
        elif sex_idx == 'male':
            df_discrete_yfs_sex_subset = df_discrete_yfs[df_discrete_yfs['Sex'] == 2]
            file_suffix = "_male.csv"
        else:
            print("Sex index error.")
            df_discrete_yfs_sex_subset = pd.DataFrame()

        df_discrete_yfs_to_calculate_sex = df_discrete_yfs_sex_subset.copy()
        df_discrete_yfs_to_calculate_sex.drop(columns=['Sex'], inplace=True)
        print(f'After sex group: {df_discrete_yfs_to_calculate_sex.shape}')

        # Save data
        df_discrete_yfs_to_calculate_sex.to_csv(yfs_data_name + file_suffix)

        # Calculate MI and p-value
        mi_norm_yfs_sex, mi_abs_yfs_sex = dm.cal_mi_skl_cluster(df_discrete_yfs_to_calculate_sex)
        pd.DataFrame(mi_norm_yfs_sex).to_csv(norm_mi_name + file_suffix)
        pd.DataFrame(mi_abs_yfs_sex).to_csv(abs_mi_name + file_suffix)
        p_value_mi_yfs_sex = dm.mi_p_val_chi2(df_discrete_yfs_to_calculate_sex)
        pd.DataFrame(p_value_mi_yfs_sex).to_csv(p_val_mi_name + file_suffix)

else:
    print(df_discrete_yfs_to_calculate.shape)
    df_discrete_yfs_to_calculate.to_csv(yfs_data_name + file_suffix)

    mi_norm_yfs, mi_abs_yfs = dm.cal_mi_skl_cluster(df_discrete_yfs_to_calculate)
    pd.DataFrame(mi_norm_yfs).to_csv(norm_mi_name + file_suffix)
    pd.DataFrame(mi_abs_yfs).to_csv(abs_mi_name + file_suffix)

    p_value_mi_yfs = dm.mi_p_val_chi2(df_discrete_yfs_to_calculate)
    pd.DataFrame(p_value_mi_yfs).to_csv(p_val_mi_name + file_suffix)
