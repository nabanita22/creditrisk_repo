import numpy as np

def create_loan_to_income(df):

    df = df.copy()

    df["loan_to_income"] = round(
        df["loan_amount"] / df["income"],
        2
    )

    df["loan_to_income"] = df["loan_to_income"].fillna(0)

    return df

#############################################

def create_delinquency_ratio(df):

    df = df.copy()

    df["delinquency_ratio"] = (
        df["delinquent_months"] * 100
        / df["total_loan_months"]
    ).round(1)

    return df


##############################################

def create_avg_dpd_per_delinquency(df):

    df = df.copy()

    df["avg_dpd_per_delinquency"] = np.where(
        df["delinquent_months"] != 0,

        (
            df["total_dpd"]
            / df["delinquent_months"]
        ).round(1),

        0
    )

    return df