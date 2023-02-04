from datetime import datetime
from iexfinance.stocks import get_historical_data

names = [
    'aapl', #Apple Inc, Sector: Information Technology
    'amzn', #Amazon.con Inc, Sector: Information Technology
    'msft', #Microsoft Corp, Sector: Information Technology
    'goog', #Alphabet Inc, Sector: Information Technology
    'xom',  #Exxon Mobil Corp, Sector: Energy
    'apc',  #Anadarko Petroleum Corp, Sector: Energy
    'cvx',  #Chevron, Sector: Energy
    'c',    #Citigroup, Sector: Financial
    'gs',   #Goldman Sachs Group, Sector: Financial
    'jpm',  #JPMorgan Chase & Co, Sector: Financial
    'aet',  #Aetna Inc, Sector: Health Care
    'jnj',  #Johnson & Johnson, Sector: Health Care
    'dgx',  #Quest Diagnostics, Sector: Health Care
    'spy',  #State Street's SPDR S&P 500 ETF.  A security that roughly tracks
            #the S&P 500, a weighted average of the stock prices of
            #500 top US companies.
    'xlf',  #State Street's SPDR Financials ETF.  A security that tracks
            #a weighted average of top US financial companies.
    'sso',  #ProShares levered ETF that roughly corresponds to twice
            #the daily performance of the S&P 500.
    'sds',  #ProShares inverse levered ETF that roughly corresponds to 
            #twice the negative daily performance of the S&P 500.  That is,
            #when the S&P 500 goes up by a dollar, this roughly goes down by 2.
    'uso',  #Exchange traded product that tracks the price of oil in the US.
]

start = datetime(2017, 1, 1)
end = datetime(2018, 9, 20)

df = get_historical_data(names, start, end,output_format='pandas')
df.columns = [' '.join(col).strip() for col in df.columns.values]
cols = [n.upper()+' close' for n in names]
df = df[cols]
df.columns = [col.split(' ')[0] for col in df.columns.values]
df.to_csv('stockprices.csv')
