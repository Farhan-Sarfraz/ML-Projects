# ============================================================
# FINSURE AI
# MODEL EVALUATOR
# PHASE 5
# ============================================================

from typing import Dict, Any

import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


class ModelEvaluator:

    def evaluate(
        self,
        model,
        X_test,
        y_test,
        feature_names=None,
    ) -> Dict[str, Any]:

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        report = classification_report(

            y_test,

            predictions,

            output_dict=True,

            zero_division=0,

        )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(

            y_test,

            predictions,

        )

        # ----------------------------------------------------
        # ROC Curve
        # ----------------------------------------------------

        fpr, tpr, thresholds = roc_curve(

            y_test,

            probabilities,

        )

        roc_auc = auc(

            fpr,

            tpr,

        )

        # ----------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------

        feature_importance = None

        if hasattr(
            model,
            "feature_importances_"
        ):

            feature_importance = pd.DataFrame({

                "Feature":
                feature_names,

                "Importance":
                model.feature_importances_,

            })

            feature_importance = (

                feature_importance

                .sort_values(

                    by="Importance",

                    ascending=False,

                )

                .reset_index(drop=True)

            )

        return {

            "classification_report":
            report,

            "confusion_matrix":
            cm,

            "roc_auc":
            roc_auc,

            "fpr":
            fpr,

            "tpr":
            tpr,

            "thresholds":
            thresholds,

            "feature_importance":
            feature_importance,

        }


# ============================================================
# END OF FILE
# ============================================================