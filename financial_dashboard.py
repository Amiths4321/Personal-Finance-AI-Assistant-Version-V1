
import streamlit as st

def display_financial_dashboard(analysis, insights):

    st.header("Financial Intelligence Dashboard")

    st.subheader("Monthly Financial Position")

    col1, col2, col3 = st.columns(3)

    with col1:
        value = analysis.get("monthly_income")
        st.metric(
            "Monthly Income",
            f"₹{value:,.0f}" if value is not None else "N/A"
        )

    with col2:
        value = analysis.get("monthly_expenses")
        st.metric(
            "Monthly Expenses",
            f"₹{value:,.0f}" if value is not None else "N/A"
        )

    with col3:
        value = analysis.get("monthly_surplus")
        st.metric(
            "Monthly Surplus",
            f"₹{value:,.0f}" if value is not None else "N/A"
        )

    st.subheader("Assets and Liabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        value = analysis.get("total_assets")
        st.metric(
            "Total Assets",
            f"₹{value:,.0f}" if value is not None else "N/A"
        )

    with col2:
        value = analysis.get("total_liabilities")
        st.metric(
            "Total Liabilities",
            f"₹{value:,.0f}" if value is not None else "N/A"
        )

    with col3:
        value = analysis.get("net_worth")
        st.metric(
            "Net Worth",
            f"₹{value:,.0f}" if value is not None else "N/A"
        )

    st.subheader("Financial Ratios")

    col1, col2 = st.columns(2)

    with col1:
        value = analysis.get("expense_ratio_percent")
        st.metric(
            "Expense Ratio",
            f"{value:.2f}%" if value is not None else "N/A"
        )

    with col2:
        value = analysis.get("debt_to_income_percent")
        st.metric(
            "Debt-to-Income Ratio",
            f"{value:.2f}%" if value is not None else "N/A"
        )

    st.subheader("Financial Insights")

    for insight in insights:
        st.write("•", insight)
