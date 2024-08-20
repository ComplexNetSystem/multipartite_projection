import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_methods as dm
from feature_engine.imputation import RandomSampleImputer
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


# Replace each missing value with a random sample from known values.
rsi = RandomSampleImputer()

# Remove columns where 80% are missing.
remove_col_nan_percent = 0.8

# In comment: data preparation, already saved in csv files used in following lines of codes.
# Remove all-NaN rows, impute NaN values.
# discrete already
depression_symptoms_df = pd.read_csv("symptoms2007.csv", index_col=0).dropna(axis=0, how='all')
depression_symptoms_df = depression_symptoms_df.loc[:, depression_symptoms_df.isin([np.nan]).mean() < remove_col_nan_percent]
depression_symptoms_df.replace([11, 12], 1, inplace=True)
depression_symptoms_df.replace([21, 22], 2, inplace=True)
depression_symptoms_df.replace([31, 32], 3, inplace=True)
depression_symptoms_df = rsi.fit_transform(depression_symptoms_df)
# depression_symptoms_df = depression_symptoms_df.astype('int')

# discrete already, but dynamic range is high
depress_df = pd.read_csv("depression2007.csv", index_col=0).dropna(axis=0, how='all')
depress_df.drop(['masennus_neliluokkainen'], axis=1, inplace=True)  # Remove duplicate depression variable


phenotype_df = pd.read_csv("phenotypes2007.csv", index_col=0).dropna(axis=0, how='all')
phenotype_df = phenotype_df.loc[:, phenotype_df.isin([np.nan]).mean() < remove_col_nan_percent]
phenotype_df = rsi.fit_transform(phenotype_df)


nmr_df = pd.read_csv("nmr2007.csv", index_col=0).dropna(axis=0, how='all')
nmr_df = nmr_df.loc[:, nmr_df.isin([np.nan]).mean() < remove_col_nan_percent]
nmr_df = rsi.fit_transform(nmr_df)


lipid_df_sample = pd.read_csv("lipidome2007.csv", index_col=0).dropna(axis=0, how='all')  # ild. "SAMPLE_NAME"
lipid_df_sample = lipid_df_sample.loc[:, lipid_df_sample.isin([np.nan]).mean() < remove_col_nan_percent]
lipid_df = lipid_df_sample.drop(columns=['SAMPLE_NAME'])  # drop "SAMPLE_NAME"

meta_pearson = ["Crea07", "Val07", "Leu07"]
lipid_pearson = ["SM.36.0", "LPC.19.0_sn1", "Cer.d18.1.18.0."]
meta_mi = ["Crea07", "XLHDLPL07", "SLDLTG07"]
lipid_mi = ["SM.40.0", "PC.38.4b", "PC.36.4b..2", ]

meta_lipid_detected = [meta_mi, lipid_mi]
# meta_lipid_detected = [meta_mi, lipid_mi]
target_vars = ["beckpisteet", "idealCVH07"]

for target in target_vars:
    fig, axes = plt.subplots(len(meta_lipid_detected), len(meta_pearson), figsize=(16, 6.5))
    if target in depress_df.columns:
        x_df = depress_df.copy()
    else:
        x_df = phenotype_df.loc[:, [target]]
    for ii in range(len(meta_lipid_detected)):
        biomarkers = meta_lipid_detected[ii]
        if ii == 0:
            color_val = "tab:blue"
        else:
            color_val = "limegreen"
        for jj in range(len(biomarkers)):
            biomarker = biomarkers[jj]
            if biomarker in nmr_df.columns:
                y_df = nmr_df.loc[:, [biomarker]]
            else:
                y_df = lipid_df.loc[:, [biomarker]]
            x_y_df = pd.concat([x_df, y_df], axis=1, join="inner")
            x_y_df_group_mean = x_y_df.groupby(target).mean()
            x_y_df_group_std = x_y_df.groupby(target).std()
            x_y_df.plot.scatter(x=target, y=biomarker, ax=axes[ii, jj], color=color_val, alpha=0.5)
            x_group_val = np.array(x_y_df_group_mean.index)
            y_group_mean = np.array(x_y_df_group_mean.iloc[:, 0])
            y_group_std = np.array(x_y_df_group_std.iloc[:, 0])

            axes[ii, jj].plot(x_group_val, y_group_mean, color="magenta", alpha=0.7, linewidth=2)
            axes[ii, jj].fill_between(x_group_val, y_group_mean-y_group_std, y_group_mean+y_group_std,
                                      color="magenta", alpha=0.2, edgecolor=None)
            axes[ii, jj].tick_params(axis='both', which='major', labelsize=16)
            axes[ii, jj].set_xlabel(target, fontsize=18)
            axes[ii, jj].set_ylabel(biomarker, fontsize=18)
    plt.tight_layout()
    plt.savefig("biomarker_identified_vs_phenotype_MI_"+target+"_0712.pdf")


"""
# beckpisteet, idealCVH07
# Metabolites: XLHDLPL07, SLDLTG07, Crea07, Val07, Leu07
# Lipids: PC.38.4b, PE.32.1, SM.40.0
x_name, y_name = "beckpisteet", "SLDLTG07"
x_df = depress_df.copy()
# x_df = phenotype_df.loc[:, [x_name]]
y_df = nmr_df.loc[:, [y_name]]
# y_df = lipid_df.loc[:, [y_name]]
x_y_df = pd.concat([x_df, y_df], axis=1, join="inner")

x = np.array(x_y_df[x_name])
y = np.array(x_y_df[y_name])

plt.scatter(x, y, alpha=0.6)
# plt.figure(figsize=(6.5, 4.5))
ax = sns.regplot(x, y, line_kws={"color": "black", "alpha": 0.6})
plt.xscale("log")
plt.yscale("log")

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
"""
