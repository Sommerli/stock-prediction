import yfinance as yf


def get_daily_marked_data(ticker, start):
    stock = yf.Ticker(ticker)
    return stock.history(start=start)[['Open', 'Close', "High", "Low"]]


def get_kw_list(ticker):
    def get_company_name(comp_name):
        if len(comp_name) > 2:
            return comp_name[0:1]
        else:
            return comp_name[0]

    stock = yf.Ticker(ticker)
    return [stock.info['symbol'], stock.info['shortName'], get_company_name(stock.info['shortName'].split(' '))]