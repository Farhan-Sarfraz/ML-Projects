# ============================================================
# FINSURE AI
# PREDICTION ENGINE
# PHASE 7
# ============================================================

from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from backend.model_manager import (
    ModelManager
)


class PredictionEngine:
    """
    Professional Loan Approval Prediction Engine

    Responsibilities
    ------------------------------------------------
    1. Load trained model
    2. Load employment encoder
    3. Transform user inputs
    4. Generate prediction
    5. Generate approval probability
    6. Generate risk level
    7. Generate explanation
    """

    def __init__(
        self,
        model_directory="models",
        artifact_directory="artifacts",
    ) -> None:

        # ----------------------------------------------------
        # Load Model
        # ----------------------------------------------------

        self.manager = ModelManager(
            model_directory
        )

        self.model = (
            self.manager.load_model()
        )

        # ----------------------------------------------------
        # Load Encoder
        # ----------------------------------------------------

        encoder_path = (
            Path(artifact_directory)
            / "employment_encoder.pkl"
        )

        self.encoder = joblib.load(
            encoder_path
        )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def calculate_risk_level(
        self,
        approval_probability: float,
    ) -> str:

        if approval_probability >= 0.80:

            return "Low Risk"

        elif approval_probability >= 0.60:

            return "Moderate Risk"

        elif approval_probability >= 0.40:

            return "High Risk"

        return "Very High Risk"

    # ========================================================
    # EXPLANATION
    # ========================================================

    def generate_explanation(
        self,
        credit_score: int,
        income: float,
        loan_amount: float,
        prediction: int,
    ) -> str:

        strengths = []
        risks = []

        if credit_score >= 700:
            strengths.append(
                "strong credit score"
            )
        else:
            risks.append(
                "low credit score"
            )

        if income >= 80000:
            strengths.append(
                "stable income"
            )
        else:
            risks.append(
                "limited income level"
            )

        if loan_amount <= 20000:
            strengths.append(
                "manageable loan amount"
            )
        else:
            risks.append(
                "high loan amount"
            )

        if prediction == 1:

            if strengths:

                return (
                    "Loan approval is likely due to "
                    + ", ".join(strengths)
                    + "."
                )

            return (
                "Loan approval appears likely "
                "based on overall applicant profile."
            )

        if risks:

            return (
                "Application risk is elevated due to "
                + ", ".join(risks)
                + "."
            )

        return (
            "Application requires additional review."
        )

    # ========================================================
    # ENCODE EMPLOYMENT STATUS
    # ========================================================

    def encode_employment_status(
        self,
        employment_status: str,
    ) -> int:

        return int(

            self.encoder.transform(
                [employment_status]
            )[0]

        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        age: int,
        income: float,
        credit_score: int,
        loan_amount: float,
        loan_term: int,
        employment_status: str,
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Encode Employment Status
        # ----------------------------------------------------

        encoded_status = (

            self.encode_employment_status(
                employment_status
            )

        )

        # ----------------------------------------------------
        # Build DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame({

            "Age": [age],

            "Income": [income],

            "Credit_Score": [
                credit_score
            ],

            "Loan_Amount": [
                loan_amount
            ],

            "Loan_Term": [
                loan_term
            ],

            "Employment_Status": [
                encoded_status
            ],

        })

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(

            self.model.predict(
                input_df
            )[0]

        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = float(

            self.model.predict_proba(
                input_df
            )[0][1]

        )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk_level = (

            self.calculate_risk_level(
                probability
            )

        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        explanation = (

            self.generate_explanation(

                credit_score,

                income,

                loan_amount,

                prediction,

            )

        )

        # ----------------------------------------------------
        # Return Result
        # ----------------------------------------------------

        return {

            "prediction":
            prediction,

            "approved":
            prediction == 1,

            "approval_probability":
            round(
                probability * 100,
                2
            ),

            "risk_level":
            risk_level,

            "explanation":
            explanation,

        }


# ============================================================
# END OF FILE
# ============================================================