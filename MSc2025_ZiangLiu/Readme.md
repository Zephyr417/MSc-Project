## BTC-USD_price.csv contains the Bitcoin price used in this project.
## CryptoLin_IE_v2.csv is the CryptoLin dataset which contains 2683 manually labeled news titles.
## OneYearNewsDataset.csv is the GDELT 2.0 news dataset which contains over 240,000 news.
## finbert_600.csv is the GDEKT 2.0 subset which contains 600 randomly selected and manually labeled news. 
## OriginalFinBERT.ipynb is the file that replicated and evaluated the performance of the orignial Finbert model
## RetrainFin600.ipynb is used to retrain FinBERT model.
## Sentiment_600_2.ipynb is the file that processed price and sentiment data, performed sentiment analysis and built trading signal.

## The project could be recreated by running these three Jupyter Notebook file in the above order. First we replicated the FinBERT performance. Then we retrained the model and applied to the GDELT 2.0 news dataset. Next we performed the sentiment analysis and built the trading signal. At last the performance were tested and evaluated.