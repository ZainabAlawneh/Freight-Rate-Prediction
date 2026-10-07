import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("data/featured_train-test.csv")

# X = df.drop(columns=['posted_rate'])
# y = df['posted_rate']

# print(df.info())


#split data:
def splitData(df):
    target = "posted_rate"
    drop_cols = [target, "date"]   # remove any column you don't want as a feature

    # date cutoffs (70% train / 15% val / 15% test by time span)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    d_min, d_max = df["date"].min(), df["date"].max()
    span = d_max - d_min
    val_start  = d_min + span * 0.70
    test_start = d_min + span * 0.85

    train = df[df["date"] <  val_start]
    val   = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test  = df[df["date"] >= test_start]

    X_train, y_train = train.drop(columns=drop_cols), train[target]
    X_val,   y_val   = val.drop(columns=drop_cols),   val[target]
    X_test,  y_test  = test.drop(columns=drop_cols),  test[target]

    print(X_train.shape, X_val.shape, X_test.shape)
    return (X_train, y_train,X_val, y_val, X_test,  y_test )


#try Cat boost model
cat_features = [ 'equipment', 'delivery', 'pickup']
X_train, y_train , X_val, y_val, X_test, y_test= splitData(df)
model = CatBoostRegressor(
    iterations=1000,         
    learning_rate=0.01,       
    depth=6,                 
    loss_function='RMSE',     
    random_seed=42,
    verbose=100               
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50, 
    plot=False
)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"CatBoostRegressor: R² Score: {r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")


# feature_importances = model.get_feature_importance()
# feature_names = X_train.columns

# sorted_indices = feature_importances.argsort()

# plt.figure(figsize=(10, 6))
# plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], align='center', color='teal')
# plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
# plt.xlabel('Feature Importance Score', fontsize=12)
# plt.ylabel('Features', fontsize=12)
# plt.title('CatBoost - Feature Importance for Freight Rate Prediction', fontsize=14)
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


#try another model:

import lightgbm as lgb
import joblib
for c in ['equipment', 'delivery', 'pickup']:
    df[c] = df[c].astype('category')
X_train, y_train , X_val, y_val, X_test, y_test= splitData(df)


model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01,
                          num_leaves=31, random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"LGMRegressor: R² Score: {r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

#save model:

# Save model
joblib.dump(model, "saved_models/lightgbm_freight_model.pkl")

print("Model saved successfully!")