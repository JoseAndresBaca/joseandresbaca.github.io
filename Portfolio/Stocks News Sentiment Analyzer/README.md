Analyzing and Visualizing the Impact of Financial News Sentiment on Stock Prices
Project Description
This project integrates sentiment analysis of financial news with stock market data to understand the influence of media sentiment on stock prices. It encompasses web scraping, natural language processing, sentiment analysis, and interactive data visualization, aiming to uncover potential correlations between news sentiment and stock performance.

Objectives and Goals
To scrape financial news headlines for key tickers (SPY, AMD, META) and perform sentiment analysis.
To correlate the sentiment scores derived from news headlines with the stock prices of these companies.
To visualize these correlations using interactive time series plots, providing insights into how news sentiment might impact stock market behavior.
Data Used
Sources: News headlines from Finviz; stock data from Yahoo Finance.
Data Description: The dataset consists of financial news headlines, their publication dates and times, and associated sentiment scores. It also includes stock prices corresponding to each news headline's date.
Analysis and Methodology
Tools and Languages: Python is the primary programming language, utilizing libraries like NLTK for natural language processing, Pandas for data manipulation, Plotly for interactive visualizations, BeautifulSoup for web scraping, and YFinance for fetching stock data.
Sentiment Analysis:
Utilization of NLTK’s VADER SentimentIntensityAnalyzer, enhanced with a customized lexicon tailored for the financial context. Keywords such as 'crash', 'rally', 'overvalued', and 'undervalued' are weighted to reflect their financial implications.
Integration of BERT (Bidirectional Encoder Representations from Transformers) model for a more nuanced sentiment analysis. This model, pre-trained on large amounts of text, offers a sophisticated approach to understanding the sentiments expressed in news headlines.
Data Scraping and Processing:
Automated web scraping from Finviz to gather recent news headlines for selected stock tickers, capturing the title, date, and time of each news item.
Parsing and formatting of date and time information to maintain consistency in the dataset.
Stock Price Data Integration:
Retrieval of historical stock prices from Yahoo Finance, aligning these with the dates of the news headlines.
Handling of non-trading days by forward-filling stock prices, ensuring a continuous dataset for analysis.
Data Integration:
Merging sentiment scores and stock price data into a singular dataset. This integration allows for a comprehensive analysis of the relationship between news sentiment and stock price fluctuations.
Visualization Strategy:
Development of interactive time series plots using Plotly, showcasing the dual trends of sentiment scores and stock prices.
Implementation of a dual-axis design, with one axis representing sentiment scores and the other representing stock prices, allowing for a direct visual comparison of the two metrics over time.
