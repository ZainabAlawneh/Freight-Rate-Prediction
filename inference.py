# inference.py
import numpy as np
import pandas as pd
import joblib
from src.data_processing import haversine

model = joblib.load("saved_models/lightgbm_freight_model.pkl")
print("Model loaded successfully.")

# medians from TRAINING data only
train_raw = pd.read_csv("data/train-test.csv")
weight_medians = train_raw.groupby("equipment")["weight"].median()

val = pd.read_csv("data/validation.csv")      # keep original row order
df = val.copy()

df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians)).abs()
df["market_index"] = df["market_index"].interpolate(method="linear").bfill().ffill()

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek

df["hav_dist"] = haversine(df["pickup_lat"], df["pickup_lon"],
                           df["delivery_lat"], df["delivery_lon"])
df["circuity"] = (df["distance"] / df["hav_dist"]).replace([np.inf, -np.inf], np.nan)

df["distance"] = np.log1p(df["distance"])    
df["hav_dist"] = np.log1p(df["hav_dist"]) 

for c in ["equipment", "delivery", "pickup"]:
    df[c] = df[c].astype("category")

X_val = df[model.feature_name_]               # same columns, same order as training

print("NaNs per column:\n", X_val.isna().sum())

pred = np.expm1(model.predict(X_val))         # log -> dollars

# fill the official template so IDs and order match exactly 
template = pd.read_csv("data/validation.csv")
template["predicted_rate"] = template["load_id"].map(dict(zip(val["load_id"], pred)))
template = template[["load_id", "predicted_rate"]]

assert len(template) == 12000
assert template["predicted_rate"].notna().all()
assert (template["predicted_rate"] > 0).all()

template.to_csv("validation_predictions.csv", index=False)
print(template.describe())