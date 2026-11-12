
from financial_analyzer import analyze_finances

sample_data = {
    "monthly_income": 85000,
    "monthly_expenses": 30000,
    "monthly_emi": 12500,
    "total_assets": 250000,
    "total_liabilities": 515000
}

result = analyze_finances(sample_data)

for key, value in result.items():
    print(f"{key}: {value}")
