import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

df = pd.read_csv("data/train-test.csv")

logging.basicConfig(filename='pipeline.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting data cleaning pipeline.")

'''
print(df.head())
print(df.shape)
print(df.info())
'''

print(df.isna().sum()) # ----> mising values

print("duplicated data= " , df.duplicated().sum())  #------ duplicated data

#print(df["weight"].describe())


#cprint(df.describe()) #---------> statistics

# Fill missing weights with the median weight of the corresponding equipment type
df['weight'] = df['weight'].fillna(
    df.groupby('equipment')['weight'].transform('median')
)
logging.info(f"Imputed 300 missing weight values using midean for same type equipment.")
print(df.isna().sum()) # ----> mising values

##########################################################################################
#visualize market_index vs time 
'''df['date'] = pd.to_datetime(df['date'])

# 2. Filter for just one specific day (e.g., '2025-01-01')
single_day = '2025-02-02'
df_one_day = df[df['date'] == pd.to_datetime(single_day)].copy()

# Set up the figure
plt.figure(figsize=(10, 5))

# 3. Plot rows sequentially for that day
plt.plot(df_one_day.index, df_one_day['market_index'], marker='o', linestyle='-', color='blue', alpha=0.7, label='Market Index')

# Highlight any missing values for that day if they exist
missing_one_day = df_one_day[df_one_day['market_index'].isnull()]
if not missing_one_day.empty:
    baseline_y = df_one_day['market_index'].min() * 0.95 if not df_one_day['market_index'].dropna().empty else 0.5
    plt.scatter(missing_one_day.index, [baseline_y] * len(missing_one_day), 
                color='red', s=80, zorder=5, label='Missing Market Index')

# Graph styling
plt.title(f'Market Index Sequence for {single_day}', fontsize=14)
plt.xlabel('Row Index (Load Order)', fontsize=12)
plt.ylabel('Market Index', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

# Show exact row numbers on the x-axis
plt.xticks(df_one_day.index)
plt.tight_layout()
plt.show()'''

df['market_index'] = df['market_index'].interpolate(method='linear')
logging.info(f"Imputed 307 missing market_index values using linear interpolation.")

print("check missing again", df.isna().sum()) # ----> mising values


#  Extract the month (returns integers from 1 to 12)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

# Extract the day of the week (returns integers from 0 to 6, where 0=Monday, 6=Sunday)
df['day_of_week'] = df['date'].dt.dayofweek
# Drop the original date column
df = df.drop(columns=['date'])

logging.info(f"Extract the date feature.")
df = df.drop(columns=['load_id'])
df = df.drop(columns=['day_of_week'])
df = df.drop(columns=['quote_signal'])
df = df.drop(columns=['pickup_lat'])
df = df.drop(columns=['pickup_lon'])
df = df.drop(columns=['delivery_lat'])
df = df.drop(columns=['delivery_lon'])
df = df.drop(columns=['delivery'])
df = df.drop(columns=['pickup'])

logging.info(f"drop load_id feature.")
df.to_csv('data/cleaned_train-test.csv', index=False)