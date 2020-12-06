import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import GoogleData
import MarkedData
import WikiData
import StockPredictor

ticker = 'TSLA'
start_date = '2018-01-01'

data = pd.concat([GoogleData.get_daily_google_data(MarkedData.get_kw_list(ticker), int(start_date.split('-')[0]), int(start_date.split('-')[1])),
                  WikiData.get_daily_wiki_data(ticker, start_date),
                  MarkedData.get_daily_marked_data(ticker, start_date)], axis=1)
data = data.dropna()
stock_data = data[['Open', 'Close', 'High', 'Low']]
wiki_data = data[["wikiviews"]]
google_data = pd.DataFrame()
for index in data.columns[0:3]:
    google_data[index] = data[index]

df = StockPredictor.indicator_calculation(stock_data, wiki_data, google_data)

print(df)


mlp_prediction = StockPredictor.predict_by_mlp(df)
target_values = mlp_prediction[1]

# plot MLP
Stock, ax = plt.subplots(figsize=(20, 10),nrows=2)
ax[0].plot(np.arange(0, len(mlp_prediction[0])), mlp_prediction[0], label='One day ahead MLP predicted price')
ax[0].plot(np.arange(0, len(target_values)), target_values, label='Actual price')
ax[0].set_xlabel('Days after first recorded day')
ax[0].set_ylabel('Price $')
ax[0].set_title('MLP predicted price vs actual price')
ax[0].legend()
ax[0].grid()

# plt SVM
svm_prediction = StockPredictor.predict_by_svm(df)
ax[1].plot(np.arange(0, len(svm_prediction)), svm_prediction, label='One day ahead SVM predicted price')
ax[1].plot(np.arange(0, len(target_values)), target_values, label='Actual price')
ax[1].set_xlabel('Days')
ax[1].set_ylabel('Price $')
ax[1].set_title('SVM predicted price vs actual price')
ax[1].legend()
ax[1].grid()

plt.legend()

plt.show()
