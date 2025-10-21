from pathlib import Path
import importlib
import zipfile
import kaggle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



creds = ''
path = Path('titanic')
dependent  = 'Survived'
continuous = ['Age', 'SibSp', 'Parch', 'LogFare',"Pclass"]
categorical = ["Sex","Embarked"]

def download_data():
    cred_path = Path('~/.kaggle/kaggle.json').expanduser()
    if not cred_path.exists():
        cred_path.parent.mkdir(exist_ok=True)
        cred_path.write_text(creds)
        cred_path.chmod(0o600)

    if not path.exists():
        import zipfile,kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)


def process_data(path):
    df = pd.read_csv(path)
    modes = df.mode().iloc[0]
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)
    return df

def spleet_to_validation_and_data(df):
    random.seed(42)
    trn_df,val_df = train_test_split(df, test_size=0.25)
    trn_df[categorical] = trn_df[categorical].apply(lambda x: x.cat.codes)
    val_df[categorical] = val_df[categorical].apply(lambda x: x.cat.codes)
    return trn_df,val_df

def seperate_target_and_features(df):
    features = df[continuous + categorical].copy()
    target = df[dependent].copy()
    return features,target

download_data()
data = process_data(path/'train.csv')
test_data = process_data(path/'test.csv')
df_trn,df_val = spleet_to_validation_and_data(data)

trn_features, trn_target = seperate_target_and_features(df_trn)
m = DecisionTreeClassifier(max_leaf_nodes=55).fit(trn_features, trn_target)

val_features,val_target = seperate_target_and_features(df_val)
preds = m.predict(val_features)
mean_absolute_error(val_target,preds)
print("mean_absolute_error: ",mean_absolute_error(val_target,preds))



