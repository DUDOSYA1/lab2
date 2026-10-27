"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df =pd.read_csv(data.csv_path, sep='\t', na_values='NA')

    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    men = df[df['Gender'] == 'Male']
    women = df[df['Gender'] == 'Female']

    men_count = len(men)
    women_count = len(women)

    men_mri_correlation = men[features].corrwith(men['MRI_Count'], method='pearson').to_dict()
    women_mri_correlation = women[features].corrwith(women['MRI_Count'],method='pearson').to_dict()

    strongest_mri_feature = ''
    strongest_value = 0
    for correlations in [men_mri_correlation, women_mri_correlation]:
        for i in features:
            if abs(correlations[i]) > strongest_value:
                strongest_value = abs(correlations[i])
                strongest_mri_feature = i

    summary = BrainCorrelationSummary(
        men_count = men_count,
        women_count = women_count,
        women_mri_correlation = women_mri_correlation,
        men_mri_correlation = men_mri_correlation,
        strongest_mri_feature = strongest_mri_feature
    )

    return summary
