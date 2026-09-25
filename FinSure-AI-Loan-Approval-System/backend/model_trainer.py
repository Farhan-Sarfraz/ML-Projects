# ============================================================
# FINSURE AI
# MODEL TRAINER
# PHASE 4
# ============================================================

from typing import Dict, Any

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class ModelTrainer:
    """
    Train and evaluate multiple loan approval models.
    """

    def __init__(self) -> None:

        self.models = {

            "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

            "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

            "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),

        }

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    def evaluate_model(
        self,
        model,
        X_test,
        y_test,
    ) -> Dict[str, Any]:

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        return {

            "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

            "precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "f1_score":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            ),
        }

    # ========================================================
    # TRAIN SINGLE MODEL
    # ========================================================

    def train_model(
        self,
        model_name,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> Dict[str, Any]:

        model.fit(
            X_train,
            y_train
        )

        metrics = (
            self.evaluate_model(
                model,
                X_test,
                y_test
            )
        )

        return {

            "model_name":
            model_name,

            "model":
            model,

            "metrics":
            metrics,

        }

    # ========================================================
    # TRAIN ALL MODELS
    # ========================================================

    def train_all_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> Dict[str, Any]:

        results = {}

        for (
            model_name,
            model
        ) in self.models.items():

            results[
                model_name
            ] = self.train_model(

                model_name=model_name,

                model=model,

                X_train=X_train,

                y_train=y_train,

                X_test=X_test,

                y_test=y_test,

            )

        return results

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    def select_best_model(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:

        best_name = max(

            results,

            key=lambda name:
            results[name][
                "metrics"
            ][
                "roc_auc"
            ]

        )

        return results[
            best_name
        ]

    # ========================================================
    # COMPLETE TRAINING PIPELINE
    # ========================================================

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> Dict[str, Any]:

        results = (
            self.train_all_models(

                X_train,

                y_train,

                X_test,

                y_test,

            )
        )

        best_model = (
            self.select_best_model(
                results
            )
        )

        return {

            "all_models":
            results,

            "best_model":
            best_model,

        }


# ============================================================
# END OF FILE
# ============================================================