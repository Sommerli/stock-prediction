import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

import GoogleData
import MarkedData
import WikiData
import StockPredictor

ticker = 'AAPL'
start_date = '2016-01-01'

data = pd.concat([GoogleData.get_daily_google_data(MarkedData.get_kw_list(ticker), int(start_date.split('-')[0]),
                                                   int(start_date.split('-')[1])),
                  WikiData.get_daily_wiki_data(ticker, start_date),
                  MarkedData.get_daily_marked_data(ticker, start_date)], axis=1)
data = data.dropna()
stock_data = data[['Open', 'Close', 'High', 'Low']]
wiki_data = data[["wikiviews"]]
google_data = pd.DataFrame()
for index in data.columns[0:3]:
    google_data[index] = data[index]


google_data = google_data.replace(0, 1)
df = StockPredictor.indicator_calculation(stock_data, wiki_data, google_data)

# Helper values and function
market_columns = ['Open', 'Close', 'High', 'Low', 'S_O', 'S_O_S', 'RSI', 'Moving Average', 'EMA']
wiki_columns = [col for col in df.columns if 'wiki' in col or 'Wiki' in col]

target_values = StockPredictor.get_actual_target_price(df)

def prediction_plotter(ax, prediction, target, title, model_type='MLP'):
    ax.plot(np.arange(0, len(prediction)), prediction, label='One day ahead predicted price')
    ax.plot(np.arange(0, len(target)), target, label='Actual price')
    ax.set_xlabel('Days after first recorded day')
    ax.set_ylabel('Price $')
    ax.set_title(model_type + ' ' + title)
    ax.legend()
    ax.grid()
    return ax


# MLP SECTION
mlp_scenarios = []
mlp_titles = []

# SCENARIO 1  only marked data
mlp_scenarios.append(StockPredictor.predict_by_mlp(df[market_columns]))
mlp_titles.append('Scenario 1 - only marked data')


# SCENARIO 2 - marked and wiki data
mlp_scenarios.append(StockPredictor.predict_by_mlp(df[market_columns + wiki_columns]))
mlp_titles.append('Scenario 2 - marked and wiki data')

# SCENARIO 3 - marked and google data
mlp_scenarios.append(StockPredictor.predict_by_mlp(df[df.columns.drop(wiki_columns)]))
mlp_titles.append('Scenario 3 - marked and google data')

# SCENARIO 4 all data
mlp_scenarios.append(StockPredictor.predict_by_mlp(df))
mlp_titles.append('Scenario 4 - all data')

# Plot mlp scenarios
mlp_plot, ax = plt.subplots(figsize=(20, 10), nrows=len(mlp_scenarios))
for i in range(0, len(mlp_scenarios)):
    ax[i] = prediction_plotter(ax[i], mlp_scenarios[i], target_values, title=mlp_titles[i])
mlp_plot.show()


# SVM SECTION
svm_scenarios = []
svm_titles = []

# SCENARIO 1  only marked data
svm_scenarios.append(StockPredictor.predict_by_svm(df[market_columns]))
svm_titles.append('Scenario 1 - only marked data')


# SCENARIO 2 - marked and wiki data
svm_scenarios.append(StockPredictor.predict_by_svm(df[market_columns + wiki_columns]))
svm_titles.append('Scenario 2 - marked and wiki data')

# SCENARIO 3 - marked and google data
svm_scenarios.append(StockPredictor.predict_by_svm(df[df.columns.drop(wiki_columns)]))
svm_titles.append('Scenario 3 - marked and google data')

# SCENARIO 4 all data
svm_scenarios.append(StockPredictor.predict_by_svm(df))
svm_titles.append('Scenario 4 - all data')

# Plot svm predictions
svm_plot, ax = plt.subplots(figsize=(20, 10), nrows=len(svm_scenarios))
for i in range(0, len(svm_scenarios)):
    ax[i] = prediction_plotter(ax[i], svm_scenarios[i], target_values, title=svm_titles[i], model_type='SVM')
svm_plot.show()


def calculate_mse(target_values, mlp_prediction, svm_prediction):
    MSE_MLP = mean_squared_error(target_values, mlp_prediction)
    MSE_VSM = mean_squared_error(target_values, svm_prediction)
    return [MSE_MLP, MSE_VSM]


def print_prediction(mlp_prediction, svm_prediction):
    print(f'MLP-predicts close price: {mlp_prediction[-1]}\nSVM-predicts close price: {svm_prediction[-1]}')


for i in range(len(mlp_scenarios)):
    print('\nScenario ' + str(i+1))
    print(f'Mean squared error MLP: {calculate_mse(target_values, mlp_scenarios[i], svm_scenarios[i])[0]}\nMean squared error SVM: {calculate_mse(target_values, mlp_scenarios[i], svm_scenarios[i])[1]}')
    print_prediction(mlp_scenarios[i], svm_scenarios[i])

