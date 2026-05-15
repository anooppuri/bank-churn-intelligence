"""
eda.py — EDA statistics module (used by Streamlit app and research paper)
"""
import pandas as pd
import numpy as np


def get_eda_stats(df: pd.DataFrame) -> dict:
    stats = {}

    # Basic shape
    stats["n_rows"]         = len(df)
    stats["n_cols"]         = len(df.columns)
    stats["missing_values"] = int(df.isnull().sum().sum())
    stats["churn_rate"]     = round(df["Exited"].mean(), 4)
    stats["churn_count"]    = int(df["Exited"].sum())
    stats["retained_count"] = int((df["Exited"] == 0).sum())

    # Numeric describe
    num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
    desc = df[num_cols].describe().round(2)
    stats["numeric_describe"] = desc.to_dict()

    # By-group churn
    stats["churn_by_geography"] = (
        df.groupby("Geography")["Exited"]
          .agg(churn_rate="mean", churned="sum", total="count")
          .round(4).to_dict(orient="index"))
    stats["churn_by_gender"] = (
        df.groupby("Gender")["Exited"]
          .agg(churn_rate="mean", churned="sum", total="count")
          .round(4).to_dict(orient="index"))
    stats["churn_by_products"] = (
        df.groupby("NumOfProducts")["Exited"]
          .agg(churn_rate="mean", churned="sum", total="count")
          .round(4).to_dict(orient="index"))
    stats["churn_by_active"] = (
        df.groupby("IsActiveMember")["Exited"]
          .agg(churn_rate="mean", churned="sum", total="count")
          .round(4).to_dict(orient="index"))

    # Mean comparison
    stats["mean_by_churn"] = (
        df.groupby("Exited")[num_cols].mean().round(2).to_dict(orient="index"))

    # Correlation
    corr_series = df[num_cols + ["Exited"]].corr()["Exited"].drop("Exited")
    stats["correlations"] = corr_series.round(4).to_dict()

    return stats
