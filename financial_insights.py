
def generate_financial_insights(analysis):
    insights = []

    income = analysis.get("monthly_income")
    expenses = analysis.get("monthly_expenses")
    emi = analysis.get("monthly_emi")
    surplus = analysis.get("monthly_surplus")
    net_worth = analysis.get("net_worth")

    if income is None:
        insights.append(
            "Monthly income information is unavailable."
        )

    if expenses is None:
        insights.append(
            "Monthly expense information is unavailable."
        )

    if emi is None:
        insights.append(
            "Monthly EMI information is unavailable."
        )

    if surplus is not None:
        if surplus > 0:
            insights.append(
                f"Monthly cash flow surplus is INR {surplus:,.2f}."
            )
        elif surplus < 0:
            insights.append(
                f"Monthly cash flow deficit is INR {abs(surplus):,.2f}."
            )
        else:
            insights.append(
                "Monthly income is fully allocated."
            )

    if net_worth is not None:
        if net_worth > 0:
            insights.append(
                f"Net worth is positive at INR {net_worth:,.2f}."
            )
        elif net_worth < 0:
            insights.append(
                f"Liabilities exceed assets by INR {abs(net_worth):,.2f}."
            )
        else:
            insights.append(
                "Assets and liabilities are equal."
            )

    return insights
