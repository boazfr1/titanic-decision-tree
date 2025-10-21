from pathlib import Path
import importlib
import zipfile
import kaggle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


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

download_data()
data = process_data(path/'train.csv')
test_data = process_data(path/'test.csv')

m = DecisionTreeClassifier(max_leaf_nodes=4).fit(trn_xs, trn_y);
