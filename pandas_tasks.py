"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.csv_path)

    row_count = len(df)

    missing_by_column = dict()
    for i in df.columns:
        missing_by_column[i] = int(row_count - df[i].count())

    adults_over_30_count = len(df[df['Age'] > 30])

    mean_age_by_pclass= df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age_by_pclass = {int(i): float(j) for i, j in mean_age_by_pclass.items()}

    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival_rate_by_pclass = {int(i): float(j) for i, j in survival_rate_by_pclass.items()}

    highest_fares = df.sort_values('Fare', ascending=False)['Fare'].tolist()[0:5]

    return TitanicSummary(
        row_count=row_count,
        missing_by_column = missing_by_column,
        adults_over_30_count = adults_over_30_count,
        mean_age_by_pclass = mean_age_by_pclass,
        survival_rate_by_pclass = survival_rate_by_pclass,
        highest_fares = highest_fares
    )
