
from financial_data_mapper import map_extracted_financial_data
from financial_analyzer import analyze_finances

extracted_data = {
    "document_type": "Unknown",
    "currency": "₹",
    "financial_items": [
        {
            "description": "Monthly salary",
            "category": "Income",
            "amount": 85000
        },
        {
            "description": "Personal loan balance",
            "category": "Debt",
            "amount": 320000
        },
        {
            "description": "Credit card interest rate",
            "category": "Debt",
            "amount": 36
        },
        {
            "description": "Emergency reserve",
            "category": "Liquidity",
            "amount": 240000
        }
    ]
}

mapped = map_extracted_financial_data(extracted_data)
analysis = analyze_finances(mapped)

print("\nMAPPED DATA")
for key, value in mapped.items():
    print(f"{key}: {value}")

print("\nFINANCIAL ANALYSIS")
for key, value in analysis.items():
    print(f"{key}: {value}")
