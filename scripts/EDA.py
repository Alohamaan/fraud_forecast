from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filterwarnings('ignore')

df = pd.read_csv('../credit_card_fraud_dataset.csv')


def fraudPlotBasedOnQuestion(df: pd.DataFrame, group: list) -> plt.plot:
    """Builds plot of the amount of frauds per passed group"""

    return df.groupby(group)['IsFraud'].agg([np.mean]).sort_values(by='mean', ascending=False).plot(kind='bar',
                                                                                                    grid=True, rot=0)


def reshapeDataFrame(df: pd.DataFrame) -> pd.DataFrame:
    """Reshapes dataframe by dividing its date by weekday, hour, month, day, also drops id columns"""

    df['TransactionTime'] = pd.to_datetime(df['TransactionDate']).dt.hour
    df['TransactionMonth'] = pd.to_datetime(df['TransactionDate']).dt.month
    df['DayOfWeek'] = pd.to_datetime(df['TransactionDate']).dt.dayofweek
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate']).dt.day
    df.drop(columns=['TransactionID'], axis=1, inplace=True)

    return df


class Location:

    def fraudLocation(df: pd.DataFrame) -> plt.plot:
        """показывает зависимость мошеннических операций от места"""

        return fraudPlotBasedOnQuestion(df=df, group=['Location'])


class DateFraud:

    def weekday(df: pd.DataFrame) -> plt.plot:
        """показывает зависимость мошеннических операций от дня недели"""

        return fraudPlotBasedOnQuestion(df=df, group=['DayOfWeek'])

    def daytime(df: pd.DataFrame) -> plt.plot:
        """показывает зависимость мошеннических операций от времени суток"""

        return fraudPlotBasedOnQuestion(df=df, group=['TransactionTime'])

    def day(df: pd.DataFrame) -> plt.plot:
        """показывает зависимость мошеннических операций от дня месяца"""

        return fraudPlotBasedOnQuestion(df=df, group=['TransactionDate'])


class Transaction:

    def transactionType(df: pd.DataFrame) -> plt.plot:
        """показывает зависимость мошеннических операций от типа операции (возврат/покупка)"""

        return fraudPlotBasedOnQuestion(df=df, group=['TransactionType'])

    def amount(df: pd.DataFrame) -> plt.plot:
        """показывает зависимость суммы операции в зависимость от типа (возврат/покупка)"""

        return df.groupby(['TransactionType'])['Amount'].agg([np.median]).sort_values(by='median',
                                                                                      ascending=False).plot(kind='bar',
                                                                                                            grid=True,
                                                                                                            rot=0)

    def frequency(df: pd.DataFrame) -> pd.Series:
        """показывает количество возвратов и покупок"""

        return df['TransactionType'].value_counts()

    def fraudsPerNonAbsoluteDate(df: pd.DataFrame) -> plt.plot:
        """показывает зависимость мошеннических операций от дня в месяце и времени"""

        return fraudPlotBasedOnQuestion(df=df, group=['TransactionDate', 'TransactionTime'])


"""я знаю, что здесь дубляж кода, который можно легко убрать, но это нужно, чтобы графики адекватно выглядели в ноутбуке"""
