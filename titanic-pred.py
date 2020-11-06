import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# df_wiki = pd.read_excel('wiki_data.xlsx')

# figure out where the missing values are
# for col in df_train.columns:
#     nb_of_missing_values = df_train[col].isna().sum()
#     print('%s number of missing values: %d' % (col, nb_of_missing_values))

# Age number of missing values: 177
# Cabin number of missing values: 687
# Embarked number of missing values: 2
# Rest have no missing values

# Figure out the most popular port to embark
# print(df_train['Embarked'].value_counts()) # answer: S 644 C 168 Q 77

# for now, we are gonna assume that those with missing values embarked at 'S', i.e. the most popular port
df_train['Embarked'] = df_train['Embarked'].fillna('S')

# Now we try to figure out the age for the missing values
# First try : just put 0 for missing age
# After trying and preprocessing the data, we also perform the same operation on the test set
df_train['Age'] = df_train['Age'].fillna(0)
df_test['Age'] = df_test['Age'].fillna(0)

# For missing values for cabin we'll also try to fill in with a ficticious cabin
df_train['Cabin'] = df_train['Cabin'].fillna('M9999')
df_test['Cabin'] = df_test['Cabin'].fillna('M9999')

# Split Title, first name, middle name, last name
last_name_rest = df_train['Name'].str.split(',', expand = True)
title_rest = last_name_rest[1].str.split('.', expand = True)
first_middle_name = title_rest[1].str.split('(', expand = True)
first_middle_name = first_middle_name[0].str.split(expand = True)
df_train['First name'] = first_middle_name[0]
df_train['Middle name'] = first_middle_name[1]
df_train['Last name'] = last_name_rest[0]
df_train['Title'] = title_rest[0]
df_train.drop(['Name'], axis = 1, inplace = True)

last_name_rest_test = df_test['Name'].str.split(',', expand = True)
title_rest_test = last_name_rest_test[1].str.split('.', expand = True)
first_middle_name_test = title_rest_test[1].str.split('(', expand = True)
first_middle_name_test = first_middle_name_test[0].str.split(expand = True)
df_test['First name'] = first_middle_name_test[0]
df_test['Middle name'] = first_middle_name_test[1]
df_test['Last name'] = last_name_rest_test[0]
df_test['Title'] = title_rest_test[0]
df_test.drop(['Name'], axis = 1, inplace = True)

df_train['Name'] = df_train['Title'] + ' ' + df_train['First name'] + ' ' + df_train['Last name']
df_train.drop(['Title', 'First name', 'Middle name', 'Last name'], axis = 1, inplace = True)
df_train.set_index('Name', inplace=True)

df_test['Name'] = df_test['Title'] + ' ' + df_test['First name'] + ' ' + df_test['Last name']
df_test.drop(['Title', 'First name', 'Middle name', 'Last name'], axis = 1, inplace = True)
df_test.set_index('Name', inplace=True)

# keeping only ticket number
df_train['Ticket'] = df_train['Ticket'].str.extract(r'(\d+)').fillna(99999).astype(int)
df_test['Ticket'] = df_test['Ticket'].str.extract(r'(\d+)').fillna(99999).astype(int)

# Now we are gonna encode our categorical features
# we need to split cabin in two categories
df_train['Cabin letter'] = df_train['Cabin'].str.split(r'(\d+)', expand = True)[0]
df_train['Cabin number'] = pd.to_numeric(df_train['Cabin'].str.split(r'(\d+)', expand = True)[1], errors='coerce')
df_train['Cabin number'] = df_train['Cabin number'].fillna(99999)
df_train.drop(['Cabin'], axis = 1, inplace = True)

df_test['Cabin letter'] = df_test['Cabin'].str.split(r'(\d+)', expand = True)[0]
df_test['Cabin number'] = pd.to_numeric(df_test['Cabin'].str.split(r'(\d+)', expand = True)[1], errors='coerce')
df_test['Cabin number'] = df_test['Cabin number'].fillna(99999)
df_test.drop(['Cabin'], axis = 1, inplace = True)

cat_features = ['Sex', 'Embarked', 'Cabin letter']

le = LabelEncoder()
for feat in cat_features:
    df_train[feat] = le.fit_transform(df_train[feat].astype('str'))
    if feat in df_test.columns:
        df_test[feat] = le.fit_transform(df_test[feat].astype('str'))


# Now are gonna split the data in labels and features
X = df_train.drop(['Survived'], axis = 1)
y = df_train['Survived']

# now we create our predictive model, not good yet, but it runs
# X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = GradientBoostingClassifier().fit(X, y)

# After the first trial, the gradient boosted classifier computes an AUC score around 0.83
# y_pred = clf.predict(X_test)
# print(roc_auc_score(y_test, y_pred))

# print(clf.predict(df_test))
# print(df_test['PassengerId'].values)
results = pd.Series(clf.predict(df_test), index = df_test['PassengerId'].values)

print(results)
# compression_opts = dict(method='zip', archive_name='out.csv')
# results.to_csv('out.zip', index=True, compression=compression_opts)


