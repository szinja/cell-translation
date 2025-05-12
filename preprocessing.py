# preprocessing/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# %%
import collections
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import seaborn as sns
# from venn import venn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def load_data(file_path):
    """Load the dataset from a given file path"""
    df = pd.read_csv(file_path)
    return df

def clean_data(df, only_tree_features):
    """Clean the dataset by handling missing values, renaming columns, etc."""
    if not only_tree_features:
        # df['histology_lesion2'] = df['histology_lesion2'].fillna('Unknown')
        # df['lesion2_sampled'] = df['lesion2_sampled'].fillna('Unknown')
        # df['LUAD_pred_subtype'] = df['LUAD_pred_subtype'].fillna('Unknown')
        # df['adjuvant_treatment_given'] = df['adjuvant_treatment_given'].fillna('NA') 
        columns_to_fill = ['histology_lesion2', 'lesion2_sampled',]
        df[columns_to_fill] = df[columns_to_fill].fillna('Unknown')
        df['LUAD_pred_subtype'] = df['LUAD_pred_subtype'].fillna('NA')
        df['adjuvant_treatment_given'] = df['adjuvant_treatment_given'].fillna('NA') # For patients who did not receive adjuvant treatment, this column is NA.
        # df['num_cycle_na.added'] = df['num_cycle_na.added'].fillna(0)


    # Add other cleaning logic here
    return df


## Dropping null columns 
def drop_columns(df, only_tree_features):
    drop = ['cruk_id',
        'tumour_id_muttable_cruk',
        'tumour_id_per_patient',
        'CHMPlatDgName_cleaned', 
        'CHMOthDgName_cleaned',
        'AdjRadStartTime_manual',
        'AdjRadEndTime_manual',
        'Recurrence_time_use', 
        'newPrim_time_use',
        'first_dfs_any_event_rec.or.new.primary',
        'first_event_during_followup',
        'Relapse_cat_new',
        'tx100']

    ### Dropping more columns based on suggestions from Dr Rob and Dr Tillman

    """Reasons (based on email):
    - Adjuvant treatment (features 26-28) is given after surgery, whereas it'd be better to use only information available at the time of surgery.
    - Features 29-35 relate to events after surgery"""

    more_columns_to_drop = ['adjuvant_treatment_YN',
                            'adjuvant_treatment_given',
                            'num_cycle_na.added','cens_os',
                            'cens_dfs',
                            'cens_dfs_any_event',
                            'dfs_time_any_event',
                            'cens_lung_event',
                            'lung_event_time',
                            'Relapse_cat']



    if not only_tree_features:
        # Dropping columns with null entries


        df = df.drop(drop, axis = 1)
        list(df.columns)
    # Dropping columns
        df = df.drop(more_columns_to_drop, axis = 1)
        list(df.columns)

    return df






def label_encoding(df, only_tree_features):

    categorical_cols = df.select_dtypes(include=['object','bool','category']).columns.to_list()
    len(categorical_cols)
    """Label encode the relevant columns"""
    labelEncodedColumns = list()
    for cat in categorical_cols:
        if len(df[cat].unique()) < 4:
            labelEncodedColumns.append(cat)
            # print(cat,'=',len(df[cat].unique()), 'Unique Values')
    len(labelEncodedColumns)


    if not only_tree_features:
        le = LabelEncoder()
        df['lesion2_sampled'] = df['lesion2_sampled'].astype(str)
        for col in labelEncodedColumns:    
            df[col] = le.fit_transform(df[col])
    return df, labelEncodedColumns




# ordinal_columns = ['pathologyTNM','pT_stage_per_patient']
def ordinal_encoding(df,only_tree_features):
    if not only_tree_features:
    # Defining the order of the categories (by Dr Robert)
        PathologyTNM_categories_order = [['IA' ,'IB' ,'IIA', 'IIB' ,'IIIA' ,'IIIB']]

    # Creating an instance of OrdinalEncoder for PathologyTNM
        encoder = OrdinalEncoder(categories=PathologyTNM_categories_order)

        df['TNMordered'] = encoder.fit_transform(df[['pathologyTNM']])

        df['TNMordered'] = df['TNMordered'].astype(int) + 1

    # Cross checking encoding
        df[['pathologyTNM','TNMordered']]

    #==================================================================================================
    # Defining the order of the categories (by Dr Robert)
        pT_stage_categories_order = [['1a' ,'1b', '2a', '2b', '3', '4']]

    # Creating an instance of OrdinalEncoder for pT stage
        encoder = OrdinalEncoder(categories=pT_stage_categories_order)
        df['pT_stage_ordered'] = encoder.fit_transform(df[['pT_stage_per_patient']])

        df['pT_stage_ordered'] = df['pT_stage_ordered'].astype(int) + 1 

    # Cross checking encoding
        df[['pT_stage_per_patient','pT_stage_ordered']]

        print('Ordinal Encoding done')

        return df




def one_hot_encoding(df,labelEncodedColumns):

    categorical_cols = df.select_dtypes(include=['object','bool','category']).columns.to_list()
    oneHot_columns = [category for category in categorical_cols if category not in labelEncodedColumns]
    # oneHot_columns.remove('num_cycle_na.added')

    df_encoded = pd.get_dummies(df, columns=oneHot_columns)
    # Replace True with 1 and False with 0
    df_encoded = df_encoded.replace({True: 1, False: 0}).astype(int)
    df = df_encoded
    
    return df


def encode_data(df,only_tree_features):
    df, label_encoded_columns = label_encoding(df,only_tree_features)
    df = ordinal_encoding(df,only_tree_features)
    df = one_hot_encoding(df,label_encoded_columns)

    return df

# # Cross checking encoding
# df[['pathologyTNM','TNMordered']].head(50)


def split_data(df, target_col):
    X = df.drop(columns=target_col)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    """Standardize the features"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


# only_tree_features = False
# clinicaldata_with_tree_features = True


# df = pyreadr.read_r("/Users/rafa/MscAi/MLCancerResearch/20221109_TRACERx421_all_patient_df.rds")
# df = pd.DataFrame(df[None])
# len(df)

# # %% [markdown]
# # ### Incorporating Tree Indices with Clinical data by Rob & Kim

# # %%
# new_df = pd.read_csv("index_values_cleaned.csv")
# new_df = new_df.rename(columns = {'Unnamed: 0':'cruk_id'})
# print(len(new_df))
# new_df.columns


# if clinicaldata_with_tree_features:
#     df = pd.merge(df,new_df, on = 'cruk_id')
#     print('Clinical Data and Tree features')

# elif only_tree_features:
#     df = pd.merge(df[['cruk_id','os_time','dfs_time']],new_df, on = 'cruk_id')
#     print('Only Tree features')
# else:
#     print('Only Clinical Data')


# # %% [markdown]
# # ### Checking Uncommon Values (Need to investigate what are these) - Eliminated from training for now
# # #### 8 New Values in tree indices by Kim which are not present in clinical data from TracerX
# # -  CRUK0223_Tumour1
# # - CRUK0586_Tumour2
# # - CRUK0620_Tumour1
# # - CRUK0030_Tumour2
# # - CRUK0704_Tumour1
# # - CRUK0372_Tumour1
# # - CRUK0881_Tumour1
# # - CRUK0555_Tumour2

# # %%
# # uncommon_values = new_df[~new_df['cruk_id'].isin(df['cruk_id'])]
# # uncommon_values['cruk_id']

# # # %% [markdown]
# # # ## Plotting null values

# # # %%
# # # Count NaN values in each column
# # null_counts = df.isnull().sum()

# # # Create a DataFrame to display column names and null counts
# # null_counts_df = pd.DataFrame({'Column': null_counts.index, 'Null Count': null_counts.values})

# # # print("\nColumns with number of NaN values:")
# # # print(null_counts_df)


# # # Filter columns with NaN values greater than 0
# # null_counts_filtered = null_counts_df[null_counts_df['Null Count'] > 0]

# # print("\nColumns with number of NaN values greater than 0:")
# # print(null_counts_filtered)

# # import matplotlib.pyplot as plt

# # # Plotting the bar chart for filtered columns
# # plt.figure(figsize=(10,5))
# # plt.bar(null_counts_filtered['Column'], null_counts_filtered['Null Count'], color='skyblue')
# # plt.xlabel('Columns')
# # plt.ylabel(' ')
# # plt.title('Missing Values')
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.show()


# # %% [markdown]
# # ### Histology lesion2 null values replaced with 'Unknown'

# # %%
# # if not only_tree_features:
# #     # df['histology_lesion2'] = df['histology_lesion2'].fillna('Unknown')
# #     # df['lesion2_sampled'] = df['lesion2_sampled'].fillna('Unknown')
# #     # df['LUAD_pred_subtype'] = df['LUAD_pred_subtype'].fillna('Unknown')
# #     # df['adjuvant_treatment_given'] = df['adjuvant_treatment_given'].fillna('NA') 
# #     columns_to_fill = ['histology_lesion2', 'lesion2_sampled',]
# #     df[columns_to_fill] = df[columns_to_fill].fillna('Unknown')
# #     df['LUAD_pred_subtype'] = df['LUAD_pred_subtype'].fillna('NA')
# #     df['adjuvant_treatment_given'] = df['adjuvant_treatment_given'].fillna('NA') # For patients who did not receive adjuvant treatment, this column is NA.
# #     # df['num_cycle_na.added'] = df['num_cycle_na.added'].fillna(0)



# # %%
# len(df.columns)

# # %% [markdown]
# # ## Dropping null columns 

# # %%
# drop = ['cruk_id',
#         'tumour_id_muttable_cruk',
#         'tumour_id_per_patient',
#         'CHMPlatDgName_cleaned', 
#         'CHMOthDgName_cleaned',
#         'AdjRadStartTime_manual',
#         'AdjRadEndTime_manual',
#         'Recurrence_time_use', 
#         'newPrim_time_use',
#         'first_dfs_any_event_rec.or.new.primary',
#         'first_event_during_followup',
#         'Relapse_cat_new',
#         'tx100']



# # %%
# if not only_tree_features:
#     # Dropping columns with null entries
#     df = df.drop(drop, axis = 1)
#     list(df.columns)
# else:
#     df = df


# # %% [markdown]
# # ### Dropping more columns based on suggestions from Dr Rob and Dr Tillman
# # 
# # Reasons (based on email):
# # 
# #     - Adjuvant treatment (features 26-28) is given after surgery, whereas it'd be better to use only information available at the time of surgery.
# # 
# #     - Features 29-35 relate to events after surgery

# # %%
# more_columns_to_drop = ['adjuvant_treatment_YN',
#  'adjuvant_treatment_given',
#  'num_cycle_na.added','cens_os',
#  'cens_dfs',
#  'cens_dfs_any_event',
#  'dfs_time_any_event',
#  'cens_lung_event',
#  'lung_event_time',
#  'Relapse_cat']

# if not only_tree_features:
#     # Dropping columns
#     df = df.drop(more_columns_to_drop, axis = 1)
# list(df.columns)

# # %%
# # Not needed for now
# # df['num_cycle_na.added'] = df['num_cycle_na.added'].fillna('0')
# # df['num_cycle_na.added'].unique()

# # %% [markdown]
# # ## Label Encoding
# # 
# #  This technique converts each value in a column to a unique numerical value. 
# # 
# #  Label encoding is suitable when the categorical data has an ordinal relationship between the categories, as it preserves the order of the categories and allows the model to learn the relationship between them. Neural networks can learn from label encoded data by treating the numerical values as continuous variables and using them as input to the network.
# #  
# #  The problem with this is that the model might assume a relationship between these categories which doesn't exist. For example, if you label encode 'Low', 'Medium', and 'High' as 1, 2, and 3, the model might assume that 'High' is twice as important as 'Low' and 'Medium' is in between.

# # %%
# categorical_cols = df.select_dtypes(include=['object','bool','category']).columns.to_list()
# len(categorical_cols)

# # %%
# df.info()

# # %%
# # for column in categorical_cols:
# #     print(f'{column} -',df[column].unique())
# #     # print(df[column].value_counts())

# #     print('==============')

# # %% [markdown]
# # Label encoding columns with less than 4 unique values is a common and effective technique for converting categorical variables into numerical variables, especially when the number of unique values is small. 

# # %%
# labelEncodedColumns = list()
# for cat in categorical_cols:
#     if len(df[cat].unique()) < 4:
#         labelEncodedColumns.append(cat)
#         print(cat,'=',len(df[cat].unique()), 'Unique Values')
# len(labelEncodedColumns)

# # %%
# if not only_tree_features:
#     le = LabelEncoder()

#     df['lesion2_sampled'] = df['lesion2_sampled'].astype(str)

#     for col in labelEncodedColumns:    
#         df[col] = le.fit_transform(df[col])

# # %% [markdown]
# # ### Label Encoding Skipped for these columns
# # Reason : High number of unique values
# # 
# # 
# # - Ethnicity = 11 Unique Values
# # - pathologyTNM = 6 Unique Values
# # - pT_stage_per_patient = 6 Unique Values
# # - histology_lesion1 = 9 Unique Values
# # - histology_lesion1_merged = 7 Unique Values
# # - histology_lesion2 = 6 Unique Values
# # - histology_multi_full = 9 Unique Values
# # - histology_multi_full_genomically.confirmed = 7 Unique Values
# # - LUAD_pred_subtype = 9 Unique Values
# # - num_cycle_na.added = 8 Unique Values
# # - Relapse_cat = 7 Unique Values
# # 
# # 
# # 

# # %% [markdown]
# # ## Ordinal Encoding Specifically for these two columns
# # 
# # Suggested by Dr Robert Noble.

# # %%

# from sklearn.preprocessing import OrdinalEncoder

# # ordinal_columns = ['pathologyTNM','pT_stage_per_patient']
# if not only_tree_features:
#     # Defining the order of the categories (by Dr Robert)
#     PathologyTNM_categories_order = [['IA' ,'IB' ,'IIA', 'IIB' ,'IIIA' ,'IIIB']]

#     # Creating an instance of OrdinalEncoder for PathologyTNM
#     encoder = OrdinalEncoder(categories=PathologyTNM_categories_order)

#     df['TNMordered'] = encoder.fit_transform(df[['pathologyTNM']])

#     df['TNMordered'] = df['TNMordered'].astype(int) + 1

#     # Cross checking encoding
#     df[['pathologyTNM','TNMordered']]

#     #==================================================================================================
#     # Defining the order of the categories (by Dr Robert)
#     pT_stage_categories_order = [['1a' ,'1b', '2a', '2b', '3', '4']]

#     # Creating an instance of OrdinalEncoder for pT stage
#     encoder = OrdinalEncoder(categories=pT_stage_categories_order)
#     df['pT_stage_ordered'] = encoder.fit_transform(df[['pT_stage_per_patient']])

#     df['pT_stage_ordered'] = df['pT_stage_ordered'].astype(int) + 1 

#     # Cross checking encoding
#     df[['pT_stage_per_patient','pT_stage_ordered']]

#     print('Ordinal Encoding done')

# # # Cross checking encoding
# # df[['pathologyTNM','TNMordered']].head(50)

# # %%
# # df.to_csv("20221109_TRACERx421_all_patient_df_Converted.csv")

# # %%


# # %% [markdown]
# # ## One-Hot Encoding
# # 

# # %%
# oneHot_columns = [category for category in categorical_cols if category not in labelEncodedColumns]
# # oneHot_columns.remove('num_cycle_na.added')
# oneHot_columns


# # %%
# df_encoded = pd.get_dummies(df, columns=oneHot_columns)
# # Replace True with 1 and False with 0
# df_encoded = df_encoded.replace({True: 1, False: 0}).astype(int)


# # %% [markdown]
# # ## Features, Target & Standardisation

# # %%
# df = df_encoded
# df
