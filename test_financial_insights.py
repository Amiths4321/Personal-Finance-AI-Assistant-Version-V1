
from financial_insights import generate_financial_insights

sample_analysis = {
    "monthly_income": 85000,
    "monthly_expenses": None,
    "monthly_emi": None,
    "monthly_surplus": None,
    "expense_ratio_percent": None,
    "debt_to_income_percent": None,
    "total_assets": 240000,
    "total_liabilities": 320000,
    "net_worth": -80000
}

insights = generate_financial_insights(sample_analysis)

print("\nFINANCIAL INSIGHTS")

for insight in insights:
    print("-", insight)
