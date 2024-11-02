from abc import ABC, abstractmethod
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from scripts.EDA import df, reshapeDataFrame

pd.set_option('display.max_columns', 500)

df = reshapeDataFrame(df=df)
encoder = OrdinalEncoder()
df['Location'] = encoder.fit_transform(df[['Location']])
df['TransactionType'] = encoder.fit_transform(df[['TransactionType']])


class Model(ABC):

    @abstractmethod
    def createTest(df: pd.DataFrame) -> tuple:
        """splits dataframe for test and reduces class imbalance"""

        x = df.drop(['IsFraud'], axis=1)
        y = df['IsFraud']
        cat_features = ['Location', 'TransactionType', 'TransactionTime', 'TransactionMonth', 'TransactionDate',
                        'DayOfWeek', 'MerchantID']
        cat_features_dict = {i: (x[i].min(), x[i].max()) for i in cat_features}

        pipeline = Pipeline(steps=[
            ('over', RandomOverSampler(random_state=52)),
        ])

        x, y = pipeline.fit_resample(x, y)

        # защита от того, что механизмы сэмплеров могут выйти за границы исходных данных
        for i in cat_features:
            x[i] = x[i].clip(lower=cat_features_dict[i][0], upper=cat_features_dict[i][1])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17, shuffle=True)

        return X_train, X_test, y_train, y_test

    @abstractmethod
    def learn(df: pd.DataFrame) -> None:
        return


class RandomForest(Model):

    def learn(df: pd.DataFrame) -> RandomForestClassifier:
        """builds Random Forest Classifier"""

        X_train, X_test, y_train, y_test = Model.createTest(df)

        clf = RandomForestClassifier(n_estimators=100, random_state=52)
        clf.fit(X_train, y_train)

        return clf
