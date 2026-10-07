import pandas as pd
from data_processing import haversine

DEC_INPUT = "data/december-chart-inputs.csv"      # original file (not modified)
DEC_OUTPUT = "data/december_prepared.csv"         # new file for inference.py



train = pd.read_csv("data/cleaned_train-test.csv")
val = pd.read_csv("data/validation.csv")
allrows = pd.concat([train, val])

df = pd.read_csv(DEC_INPUT).drop(columns=["predicted_rate"], errors="ignore")

# ---------------------------------------------------------------
# Part 1: columns the December file is missing
# ---------------------------------------------------------------
def city_coords(city):
    r = allrows[allrows["pickup"] == city].iloc[0]
    return r["pickup_lat"], r["pickup_lon"]

df["pickup_lat"], df["pickup_lon"] = city_coords(df["pickup"].iloc[0])
df["delivery_lat"], df["delivery_lon"] = city_coords(df["delivery"].iloc[0])

# market_index / quote_signal: daily average of December rows in validation.csv
val["date"] = pd.to_datetime(val["date"])
daily = (val[val["date"] >= "2025-12-01"]
         .groupby("date")[["market_index", "quote_signal"]].mean())
df["date"] = pd.to_datetime(df["date"])
df = df.join(daily, on="date")

df["load_id"] = [f"DEC-{i:06d}" for i in range(1, len(df) + 1)]
cols = ["load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
        "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
        "date", "market_index", "quote_signal"]
df = df[cols]


# date features
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

# # haversine distance + circuity
# df["hav_dist"] = haversine(df.pickup_lat, df.pickup_lon,
#                            df.delivery_lat, df.delivery_lon)
# df["circuity"] = df["distance"] / df["hav_dist"]

# # drop the same columns as data_processing.py
# df = df.drop(columns=["month", "pickup_lat", "pickup_lon",
#                       "delivery_lat", "delivery_lon"])

# negative weights -> abs
df["weight"] = df["weight"].abs()

# (posted_rate is the target, so nothing to transform for December)

# assert df.isna().sum().sum() == 0, "NaN found in December data"

df.to_csv(DEC_OUTPUT, index=False)



print(f"Saved {DEC_OUTPUT} ({len(df)} rows)")
print(df.head())