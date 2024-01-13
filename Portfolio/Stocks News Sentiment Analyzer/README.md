# Analyzing and Visualizing the Impact of Financial News Sentiment on Stock Prices

## Project Description
This project integrates sentiment analysis of financial news with stock market data to understand the influence of media sentiment on stock prices. It encompasses web scraping, natural language processing, sentiment analysis, and interactive data visualization, aiming to uncover potential correlations between news sentiment and stock performance.

## Objectives and Goals
- To scrape financial news headlines for key tickers (SPY, AMD, META) and perform sentiment analysis.
- To correlate the sentiment scores derived from news headlines with the stock prices of these companies.
- To visualize these correlations using interactive time series plots, providing insights into how news sentiment might impact stock market behavior.

## Data Used
### Sources
- News headlines from Finviz.
- Stock data from Yahoo Finance.

### Data Description
The dataset consists of financial news headlines, their publication dates and times, and associated sentiment scores. It also includes stock prices corresponding to each news headline's date.

## Analysis and Methodology
### Tools and Languages
Python is the primary programming language, utilizing libraries like NLTK for natural language processing, Pandas for data manipulation, Plotly for interactive visualizations, BeautifulSoup for web scraping, and YFinance for fetching stock data.

### Sentiment Analysis
- **VADER SentimentIntensityAnalyzer**: Utilized with a customized lexicon tailored for the financial context. Keywords such as 'crash', 'rally', 'overvalued', and 'undervalued' are weighted to reflect their financial implications.
- **BERT (Bidirectional Encoder Representations from Transformers)**: Integrated for a more nuanced sentiment analysis. This model, pre-trained on large amounts of text, offers a sophisticated approach to understanding the sentiments expressed in news headlines.

### Data Scraping and Processing
- Automated web scraping from Finviz to gather recent news headlines for selected stock tickers, capturing the title, date, and time of each news item.
- Parsing and formatting of date and time information to maintain consistency in the dataset.

### Stock Price Data Integration
- Retrieval of historical stock prices from Yahoo Finance, aligning these with the dates of the news headlines.
- Forward-filling stock prices for non-trading days to ensure a continuous dataset for analysis.

### Data Integration
Merging sentiment scores and stock price data into a singular dataset. This integration allows for a comprehensive analysis of the relationship between news sentiment and stock price fluctuations.

### Visualization Strategy
- Development of interactive time series plots using Plotly, showcasing the dual trends of sentiment scores and stock prices.
- Implementation of a dual-axis design, with one axis representing sentiment scores and the other representing stock prices, allowing for a direct visual comparison of the two metrics over time.

## Results

The sentiment analysis performed on SPY (S&P 500 ETF) involved examining financial news headlines and their subsequent influence on stock prices from November 21, 2023, to December 8, 2023.

| Date       | News                                                       | Positivity Score | Stock Price |
|------------|------------------------------------------------------------|------------------|-------------|
| 2023-12-08 | Want to Grow Your Portfolio to $1 Million? Here's How...   | 0.0772           | 458.230011  |
| 2023-12-07 | 3 reasons a 'boring' December could get interesting for... | 0.4019           | 458.230011  |
| 2023-12-06 | READ: ETF of the Week: Know What's Under the Hood          | 0.0000           | 454.760010  |
| 2023-12-04 | What's the Best ETF to Buy Right Now?                      | 0.6369           | 456.690002  |
| 2023-12-03 | Which Countries Have the Highest Tariffs?                  | 0.0000           | 459.100006  |
| 2023-12-01 | S&P 500 Average Return                                      | 0.0000           | 459.100006  |
| 2023-11-30 | Why Christmas may have come early this year for...         | 0.0000           | 456.399994  |
| 2023-11-27 | How To Earn $500 A Month From SPY Stock                    | 0.0000           | 454.480011  |
| 2023-11-24 | 12 Best Stocks In Each Sector                               | 0.6369           | 455.299988  |
| 2023-11-22 | Why Thanksgiving week is typically bullish for...          | 0.0000           | 455.019989  |
| 2023-11-21 | Investors Avoiding SPYs Concentrated Holdings...           | -0.5719          | 453.269989  |

The table reflects the varying levels of positivity in news sentiment alongside the stock prices, indicating potential influences of public sentiment on market fluctuations.

### Observations from the Sentiment Analysis:
- **Volatile Sentiment**: The sentiment scores varied significantly over the observed period, reflecting a dynamic news environment.
- **Stock Price Trend**: Despite fluctuations in sentiment, the stock price of SPY showed a general upward trend with interim volatility.

### Notable Correlations:
- **Positive Correlation**: On days like December 4, 2023, a high sentiment score was mirrored by a rise in the stock price, suggesting moments of positive correlation.
- **Negative Sentiment and Stock Price Dip**: On November 21, 2023, a notably low sentiment score preceded a decrease in stock price, which could imply a reactionary dip.

### Discrepancies in Correlation:
- **Inconsistent Correlation**: There were several instances where the sentiment did not align with stock price movements. For example, despite a high sentiment score on November 24, 2023, the stock price did not exhibit a corresponding increase, indicating that other market forces were at play.

### Analysis of Extremes:
- **Impact of Extreme Sentiment**: The analysis specifically looked at the extreme values of sentiment to see if they had a pronounced effect on stock prices. The data showed that while extreme positive sentiment scores sometimes coincided with price increases, the reverse was not always true for negative sentiment.

## Conclusions

The investigation into SPY's news sentiment and stock price relationship indicates a complex interaction. While there are instances where sentiment appears to directly affect stock prices, these are not consistently reliable due to the influence of various other market factors. The findings underscore the need for a multifaceted approach when considering sentiment as an indicator for stock price movements.

