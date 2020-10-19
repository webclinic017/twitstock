import stockdatafetch1
     
stockdata = stockdatafetch1.StockDataFetch(['IBM','AMZN','AAPL'],'Intraday')


print(stockdata.outputsize)
