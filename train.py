import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("data/cleaned_train-test.csv")

X = df.drop(columns=['posted_rate'])
y = df['posted_rate']
print(df.info())

cat_features = [ 'equipment']

#split data:

tscv = TimeSeriesSplit(n_splits=3)

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print(f"Fold {fold+1}: Train size = {len(X_train)}, Test size = {len(X_test)}")
    


    model = CatBoostRegressor(
        iterations=1000,         
        learning_rate=0.05,       
        depth=6,                 
        loss_function='RMSE',     
        random_seed=42,
        verbose=100               
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50, 
        plot=False
    )

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")


    feature_importances = model.get_feature_importance()
    feature_names = X_train.columns

    sorted_indices = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], align='center', color='teal')
    plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('CatBoost - Feature Importance for Freight Rate Prediction', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()