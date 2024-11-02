import re
import matplotlib.pyplot as plt
import pandas as pd

import shap
import tqdm

from alibi.explainers import AnchorTabular
from alibi.utils import gen_category_map


class Explanation:

    def buildExplanationPlot(model, X_train: pd.DataFrame, X_test: pd.DataFrame, sampleSize=100):
        X_train = shap.sample(X_train, sampleSize)
        X_test = shap.sample(X_test, sampleSize)
        features = X_train.columns.tolist()

        ex = shap.KernelExplainer(model.predict, X_train)
        shap_values = ex.shap_values(X_test)
        expected_value = ex.expected_value
        shap_explained = shap.Explanation(shap_values, feature_names=features)

        shap.summary_plot(shap_explained, X_test, feature_names=features)

        shap.plots.bar(shap_values=shap_explained, max_display=len(features))

        shap.plots.decision(base_value=expected_value, shap_values=shap_values,
                            feature_names=features)
        return

    def buildAnchorTabular(model, X_train: pd.DataFrame, X_test: pd.DataFrame, slice=100) -> plt.plot:
        features = X_train.columns.tolist()
        cat_map = gen_category_map(X_train)
        X_train = shap.sample(X_train, slice).to_numpy()
        X_test = shap.sample(X_test, slice).to_numpy()

        ex = AnchorTabular(model.predict, feature_names=features, categorical_names=cat_map)
        ex.fit(X_train)

        res = {x: 0 for x in features}
        for i in tqdm.tqdm(range(slice)):

            explanation = ex.explain(X=X_test[i], threshold=0.98)
            temp = explanation.data['anchor']

            if len(temp) > 0:
                for x in temp:
                    res[re.search('[a-z]+', x, flags=re.IGNORECASE).group()] += 1

        anchorAppearences = sorted(res.items(), key=lambda x: x[1], reverse=True)
        data = pd.DataFrame(data=anchorAppearences, columns=['Feature', 'Score'])
        data.plot(kind='barh', y='Score', x='Feature', grid=True)
        plt.show()

        return

