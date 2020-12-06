import pandas as pd
from pytrends import dailydata
import os.path
from os import path
import pageviewapi as wik
from datetime import date, datetime as dt, timedelta
import pandas as pd
import numpy as np

#imports from sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error


def indicator_calculation(stock_data, wiki_data, google_data):
    df_google = google_data
    df_stock = stock_data
    df_wiki = wiki_data
    out_df = pd.concat([stock_data, wiki_data], axis=1)

    def Stochastic_Oscillator(data):
        df = pd.DataFrame()
        df["L14"] = df_stock["Low"].rolling(window=14).min()
        df["H14"] = df_stock["High"].rolling(window=14).max()
        df["%K"] = ((df_stock["Close"] - df["L14"]) / (df["H14"] - df["L14"])) * 100
        df["%D"] = df["%K"].rolling(window=3).mean()
        fast_stocastic = df["%K"]
        slow_stochastic = df["%D"]

        # lalala = pd.concat([df["%K"], df["%D"]], axis = 1)
        stochastic_oscillator = pd.concat([fast_stocastic, slow_stochastic], axis=1)

        # Kan legge inn ein mean på 3 dagar dersom det er nødvendig for modellen for å få ein slow stochastic indicator
        return stochastic_oscillator

    def RSI(data, time_window=14):
        diff = data.diff(1).dropna()

        up_change = 0 * diff
        down_change = 0 * diff

        up_change[diff > 0] = diff[diff > 0]
        down_change[diff < 0] = diff[diff < 0]

        up_change_avg = up_change.ewm(com=time_window - 1, min_periods=time_window).mean()
        down_change_avg = down_change.ewm(com=time_window - 1, min_periods=time_window).mean()

        rs = abs(up_change_avg / down_change_avg)
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def William_R(data):
        df = pd.DataFrame()
        df["L14"] = df_stock["Low"].rolling(window=14).min()
        df["H14"] = df_stock["High"].rolling(window=14).max()
        df["William_R"] = ((df["H14"] - df_stock["Close"]) / (df["H14"] - df["L14"])) * -100
        William_R = df["William_R"]

        # https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r
        # https://www.investopedia.com/terms/w/williamsr.asp
        return William_R

    def Moving_Average(data, time_window=3):
        data_rolling = data.rolling(window=time_window).mean()
        return data_rolling

    def EMA(data):
        ema = data.ewm(span=3,
                       adjust=False).mean()  # https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#stats-moments-exponentially-weighted

        return ema

    def Disparity(data):
        disparity = (data / Moving_Average(data)) * 100
        return disparity

    def Momentum_1(data):
        momentum1 = (data / data.shift(5)) * 100
        return momentum1

    def Rate_Of_Change(data):
        momentum2 = (data / data.shift(5)) * 100
        ROC = (data / momentum2) * 100
        return ROC

    out_df[["S_O", "S_O_S"]] = Stochastic_Oscillator(df_stock[["High", "Low", "Close"]])
    out_df["RSI"] = RSI(df_stock["Close"], 14)
    out_df["William_R"] = William_R(df_stock[["High", "Low", "Close"]])
    out_df["Moving Average"] = Moving_Average(df_stock["Close"], 3)
    out_df["EMA"] = EMA(df_stock["Close"])

    out_df["Wiki RSI"] = RSI(df_wiki["wikiviews"], 14)
    out_df["Wiki MA"] = Moving_Average(df_wiki["wikiviews"], 14)
    out_df["Wiki EMA"] = EMA(df_wiki["wikiviews"])
    out_df["Wiki Dis"] = Disparity(df_wiki["wikiviews"])
    out_df["Wiki Mom_1"] = Momentum_1(df_wiki["wikiviews"])
    out_df["Wiki ROC"] = Rate_Of_Change(df_wiki["wikiviews"])

    ######## ADDING GOOGLE DATA - LARS
    funclist = [RSI, Moving_Average, EMA, Disparity]  # Momentum1 and Rate_of_change does not play well with google data
    for index, function in [(index, function) for index in df_google.columns for function in funclist]:
        out_df[index + (' ') + str(function.__name__)] = function(df_google[index])

        ##########
    return out_df.dropna()


# returns scaled target price and the scaler
def get_scaled_target_price(df):
    # Setup target
    target_price = np.array(df['Close'].shift(-1).dropna())
    target_price_scaler = StandardScaler()
    target_price_scaler.fit(target_price.reshape(-1, 1))
    scaled_target_price = target_price_scaler.transform(target_price.reshape(-1, 1))
    return [scaled_target_price, target_price_scaler]


def get_pca_data(df):
    # Setup PCA-scaler
    pca_scaler = StandardScaler()  # StandardScaler scales columnwise
    pca_scaler.fit(df)
    pca_scaler_data = pca_scaler.transform(df)

    # Setup PCA
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(pca_scaler_data)
    pca_data = pca.transform(pca_scaler_data)
    return pca_data


# returns array with two elements, index 0: predicted values, index 1: target values
def predict_by_mlp(df):
    pca_data = get_pca_data(df)
    temp = get_scaled_target_price(df)
    scaled_target_price = np.array(temp[0]).ravel()
    input_X = np.array(pca_data[:-1])

    X_train, X_test, y_train, y_test = train_test_split(input_X, scaled_target_price, random_state=1, test_size=0.2)

    mlp_regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    mlp_prediction = mlp_regr.predict(input_X)
    return [temp[1].inverse_transform(mlp_prediction), temp[1].inverse_transform(scaled_target_price)]


def predict_by_svm(df):
    pca_data = get_pca_data(df)
    temp = get_scaled_target_price(df)
    scaled_target_price = np.array(temp[0]).ravel()
    input_X = np.array(pca_data[:-1])

    X_train, X_test, y_train, y_test = train_test_split(input_X, scaled_target_price, random_state=1, test_size=0.2)

    scaled_svm = svm.SVR(gamma='auto')
    scaled_svm.fit(X_train, y_train)
    return temp[1].inverse_transform(scaled_svm.predict(pca_data))
