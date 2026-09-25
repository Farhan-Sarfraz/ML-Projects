# ============================================================
# FINSURE AI
# BUSINESS ANALYTICS ENGINE
# PHASE 10
# ============================================================

from typing import Dict, Any


class AnalyticsEngine:
    """
    Converts raw model predictions into
    business-friendly loan insights.
    """

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    def generate_recommendation(
        self,
        approved: bool,
        probability: float,
    ) -> str:

        if approved and probability >= 90:

            return (
                "Proceed with loan approval."
            )

        if approved:

            return (
                "Approve with standard review."
            )

        if probability >= 40:

            return (
                "Manual review recommended."
            )

        return (
            "Reject application."
        )

    # ========================================================
    # KEY FACTORS
    # ========================================================

    def generate_key_factors(
        self,
        credit_score,
        income,
        loan_amount,
    ):

        factors = []

        if credit_score >= 700:

            factors.append(
                "Strong credit score"
            )

        else:

            factors.append(
                "Weak credit score"
            )

        if income >= 80000:

            factors.append(
                "Stable income"
            )

        else:

            factors.append(
                "Limited income level"
            )

        if loan_amount <= 20000:

            factors.append(
                "Low loan burden"
            )

        else:

            factors.append(
                "High loan burden"
            )

        return factors

    # ========================================================
    # BUILD REPORT
    # ========================================================

    def build_report(
        self,
        prediction_result: Dict[str, Any],
        age,
        income,
        credit_score,
        loan_amount,
    ) -> Dict[str, Any]:

        recommendation = (

            self.generate_recommendation(

                prediction_result[
                    "approved"
                ],

                prediction_result[
                    "approval_probability"
                ],

            )

        )

        key_factors = (

            self.generate_key_factors(

                credit_score,

                income,

                loan_amount,

            )

        )

        return {

            "decision":

            "Approved"

            if prediction_result[
                "approved"
            ]

            else

            "Rejected",

            "approval_probability":

            prediction_result[
                "approval_probability"
            ],

            "risk_level":

            prediction_result[
                "risk_level"
            ],

            "recommendation":

            recommendation,

            "key_factors":

            key_factors,

            "explanation":

            prediction_result[
                "explanation"
            ],

        }


# ============================================================
# END OF FILE
# ============================================================