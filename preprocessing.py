# Usage
import pandas as pd
def preprocess_data(df, RMV=[]):
    # Standardize column names
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)

    # Ensure ID column exists and is properly named
    if "id" in df.columns:
        df.rename(columns={"id": "ID"}, inplace=True)
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))  # Create unique ID column if missing
    
    """ Cleans and scales dataset """
    FEATURES = [c for c in df.columns if not c in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
# Separate features into categorical and numerical
    CATS = [c for c in FEATURES if df[c].dtype == "object"]
    NUMS = [c for c in FEATURES if df[c].dtype != "object"]

# LABEL ENCODE CATEGORICAL FEATURES
    print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
    for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            print(f"{c}, ",end="")
            df[c],_ = df[c].factorize()
            df[c] -= df[c].min()
            df[c] = df[c].astype("int32")
            
        # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
        else:
            if df[c].dtype=="float64":
                df[c] = df[c].astype("float32")
            if df[c].dtype=="int64":
                df[c] = df[c].astype("int32")
        df[c] = df[c].fillna(0)
    # Label encode categorical features
    for c in CATS:
        print(f"{c}, ", end="")
        df[c] = df[c].fillna(0)  # Fill missing values
        df[c] = df[c].astype("category").cat.codes.astype("int32")  # Encode to int32

    # Reduce memory usage for numerical features
    for c in NUMS:
        df[c] = df[c].fillna(df[c].median())  # Ensure missing values are handled
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64":
            df[c] = df[c].astype("int32")
    return df, CATS, NUMS
