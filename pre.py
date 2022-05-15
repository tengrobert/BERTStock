import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2015-12-31', end='2019-12-31')

price = pd.read_csv('./SPY.csv', index_col='Date')
price.index = pd.to_datetime(price.index)
price['Next_close'] = price['Close'].shift(periods=-3)
price['label'] = ((price['Next_close'] - price['Open']) > 0).astype(int)
df = pd.read_csv('./prep_tweet.csv', index_col=0)
df['year'] = df['year'].astype(int).astype(str)
df['month'] = df['month'].astype(int).astype(str)
df['day'] = df['day'].astype(int).astype(str)
df['date'] = df['year'] + '-' + df['month'] + '-' + df['day']
df['date'] = pd.to_datetime(df['date'])
df = df[~df['date'].isin(holidays)]
df = df[df['date'] >= '2016-01-01']
df['text'] = df['text'].str.strip()
df = df[df['text'].str[:3] != 'rt ']
df = df[df['text'] != '']
df = df[~df['text'].isnull()]

def f(x):
    if x.date().strftime('%Y-%m-%d') in price.index:
        return price['label'][x.date().strftime('%Y-%m-%d')]
    else:
        return -1

df['label'] = df['date'].apply(f)
df = df[df['label'] != -1]
df = df[['date', 'text', 'label']]
df = df.reset_index(drop=True)
df.to_csv('./data.csv')