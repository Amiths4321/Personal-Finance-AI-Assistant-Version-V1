
def analyze_finances(data):
    income = data.get("monthly_income")
    expenses = data.get("monthly_expenses")
    emi = data.get("monthly_emi")
    assets = data.get("total_assets")
    liabilities = data.get("total_liabilities")

    def safe_subtract(*values):
        if any(value is None for value in values):
            return None
        return sum(values[:1]) - sum(values[1:])

    monthly_surplus = safe_subtract(income, expenses, emi)

    expense_ratio = (
        round(expenses / income * 100, 2)
        if income is not None and income > 0 and expenses is not None
        else None
    )

    debt_to_income = (
        round(emi / income * 100, 2)
        if income is not None and income > 0 and emi is not None
        else None
    )

    net_worth = safe_subtract(assets, liabilities)

    return {
        "monthly_income": income,
        "monthly_expenses": expenses,
        "monthly_emi": emi,
        "monthly_surplus": monthly_surplus,
        "expense_ratio_percent": expense_ratio,
        "debt_to_income_percent": debt_to_income,
        "total_assets": assets,
        "total_liabilities": liabilities,
        "net_worth": net_worth
    }
