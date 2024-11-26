#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# Parameters

C = 0.5
n_splits = 5
output_file = 'model_.bin'


# Reading Data

df = pd.read_csv('student-data.csv')

# Data Cleaning


df.columns = df.columns.str.lower().str.replace(' ', '_') # replacing spaces with underscores

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower()


df = df.rename(columns={'have_you_ever_had_suicidal_thoughts_?': 'suicidal_thoughts'}) # renaming the long heading to something short

df.suicidal_thoughts = (df.suicidal_thoughts == 'yes').astype(int)
df.family_history_of_mental_illness = (df.family_history_of_mental_illness == 'yes').astype(int)
df.depression = (df.depression == 'yes').astype(int)


# Splitting Data

df_full_train, df_test = train_test_split(df, test_size=0.2, stratify=df['depression'], random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.3, stratify=df_full_train['depression'], random_state=42)

y_train = df_train.depression.values
y_val = df_val.depression.values
y_test = df_test.depression.values


# To avoid data leakage, I have defined a columns array that excludes our target variable y (depression) when training



numerical = ['age', 'academic_pressure', 'study_satisfaction', 
             'suicidal_thoughts', 'study_hours', 'financial_stress', 
             'family_history_of_mental_illness']

categorical = ['gender', 'sleep_duration', 'dietary_habits']


len(df_full_train), len(df_train), len(df_val), len(df_test)

# Validting the model



def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=5000, solver='liblinear')
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# K-Fold Validation

# - Since our dataset is fairly small in size we will use k-fold for cross validation before training our final model
# - Also finding the best regularization parameter for our model


print(f"Running K-Fold validation with folds: {n_splits} and C: {C}")

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)


scores = []
fold = 0

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.depression.values
    y_val = df_val.depression.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f"AUC on fold {fold} is {auc}")
    fold += 1

print("Validation Results:")
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
print("\n")


# ## Training the Final model

dv, model = train(df_full_train, df_full_train.depression.values, C=C)
y_pred = predict(df_test, dv, model)

y_pred_binary = (y_pred >= 0.5).astype(int) 


print("\nStats of Final Model:")
print(f"Precision: {precision_score(y_test, y_pred_binary)}")
print(f"Recall: {recall_score(y_test, y_pred_binary)}")
print(f"AUC: {roc_auc_score(y_test, y_pred_binary)}")


# ## Save the model



with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"The model is saved to this output file: {output_file}")
