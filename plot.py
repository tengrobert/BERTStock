import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./pred3.csv')
df_oneday = df.groupby('date').mean()
df_oneday['day_label'] = (df_oneday['predicted_labels'] >= 0.5).astype(int)
df_oneday.index = pd.to_datetime(df_oneday.index)
# true_label = df_oneday['label'].values
# pred_label = df_oneday['day_label'].values
# print(((true_label == pred_label).astype(int)).sum() / len(true_label))
up = df_oneday[df_oneday['day_label'] == 1]
down = df_oneday[df_oneday['day_label'] == 0]
price = pd.read_csv('./SPY.csv', index_col='Date')
price.index= pd.to_datetime(price.index)
price = price.reindex(df_oneday.index)
plt.plot(price['Close'], label='Close Price')
plt.scatter(up.index, price['Close'][up.index], marker='^', c='g', label='Positive')
plt.scatter(down.index, price['Close'][down.index], marker='v', c='r', label='Negative')
plt.legend(loc='upper left')
plt.show()