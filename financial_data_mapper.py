
def map_extracted_financial_data(extracted_data):
    items = extracted_data.get("financial_items", [])

    result = {
        "monthly_income": 0,
        "monthly_expenses": 0,
        "monthly_emi": 0,
        "total_assets": 0,
        "total_liabilities": 0,
        "interest_rates": [],
        "unclassified_items": []
    }

    for item in items:
        description = item.get("description", "").lower()
        category = item.get("category", "").lower()
        amount = item.get("amount")

        if not isinstance(amount, (int, float)):
            result["unclassified_items"].append(item)
            continue

        # Income
        if category == "income":
            if "monthly" in description or "salary" in description:
                result["monthly_income"] += amount
            else:
                result["unclassified_items"].append(item)

        # Expenses and EMIs
        elif category == "expense":
            if "emi" in description:
                result["monthly_emi"] += amount
            elif "monthly" in description:
                result["monthly_expenses"] += amount
            else:
                result["unclassified_items"].append(item)

        # Debt
        elif category == "debt":
            if "interest rate" in description or "rate" in description:
                result["interest_rates"].append({
                    "description": item.get("description"),
                    "rate_percent": amount
                })
            elif "balance" in description or "outstanding" in description:
                result["total_liabilities"] += amount
            else:
                result["unclassified_items"].append(item)

        # Assets and liquidity
        elif category in ("asset", "liquidity"):
            result["total_assets"] += amount

        else:
            result["unclassified_items"].append(item)

    return result
