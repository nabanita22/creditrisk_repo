class ResidenceTypeImputer:

    def fit(self, df):

        self.city_mode = (
            df.groupby("city")["residence_type"]
            .agg(lambda x: x.mode()[0])
        )

        self.global_mode = (
            df["residence_type"].mode()[0]
        )

    def transform(self, df):

        df = df.copy()

        fill_values = (
            df["city"]
            .map(self.city_mode)
            .fillna(self.global_mode)
        )

        df["residence_type"] = (
            df["residence_type"]
            .fillna(fill_values)
        )

        return df


####################################################

def cap_processing_fee(df):

    df = df.copy()

    mask = (
        df["processing_fee"] / df["loan_amount"]
    ) > 0.03

    df.loc[mask, "processing_fee"] = (
        0.03 * df.loc[mask, "loan_amount"]
    )

    return df


####################################################

def cap_gst(df):

    df = df.copy()

    mask = (
        df["gst"] / df["loan_amount"]
    ) > 0.20

    df.loc[mask, "gst"] = (
        0.20 * df.loc[mask, "loan_amount"]
    )

    return df


####################################################

def cap_net_disbursement(df):

    df = df.copy()

    mask = (
        df["net_disbursement"] > df["loan_amount"]
    )

    df.loc[mask, "net_disbursement"] = (
        df.loc[mask, "loan_amount"]
    )

    return df


####################################################

class CategoryCleaner:

    def __init__(self):

        self.mapping = {
            "loan_purpose": {
                "Personaal": "Personal"
            }
        }

    def transform(self, df):

        df = df.copy()

        for col, replace_dict in self.mapping.items():

            df[col] = df[col].replace(replace_dict)

        return df