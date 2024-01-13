
"""
José Andrés Baca Salgado
bacasalgado@gmail.com
12/15/2023
"""

# Importaciones necesarias
import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import yfinance as yf

# Descargar el léxico VADER y configurar BERT para el análisis de sentimientos
nltk.download('vader_lexicon', quiet=True)
nlp_bert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Definición de URLs y tickers
finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ["SPY", "AMD", "META"]

# Función para parsear fechas
def parse_date(date_string):
    if date_string == 'Today':
        return datetime.now().strftime('%Y-%m-%d')
    elif date_string == 'Yesterday':
        return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        try:
            return datetime.strptime(date_string, '%b-%d-%y').strftime('%Y-%m-%d')
        except ValueError:
            return None

# Web scraping para obtener las tablas de noticias de Finviz
news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={"user-agent": "my-app"})
    try:
        response = urlopen(req)
        html = BeautifulSoup(response, features="html.parser")
        news_table = html.find(id="news-table")
        news_tables[ticker] = news_table
    except Exception as e:
        print(f"Error al procesar el ticker {ticker}: {e}")

# Parseo de datos de noticias
parsed_data = []
for ticker, news_table in news_tables.items():
    if news_table is not None:
        for row in news_table.findAll("tr"):
            a_tag = row.find("a")
            td_text = row.td.text.strip().split(" ")
            date = parse_date(td_text[0]) if len(td_text) > 1 else None
            time = td_text[-1]
            if a_tag and date:
                title = a_tag.text
                parsed_data.append([ticker, date, time, title])

# Creación del DataFrame de noticias
df_news = pd.DataFrame(parsed_data, columns=["ticker", "date", "time", "title"])
df_news['date'] = pd.to_datetime(df_news['date'])

# Personalización del léxico de VADER
vader = SentimentIntensityAnalyzer()
new_words = {
    'crash': -4.0,
    'rally': 4.0,
    'overvalued': -3.0,
    'undervalued': 3.0
}
vader.lexicon.update(new_words)

# Análisis de sentimiento con VADER
df_news['compound'] = df_news['title'].apply(lambda title: vader.polarity_scores(title)['compound'])

# Función para obtener los precios de las acciones
def get_stock_prices(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

# Obtener precios de acciones y agregarlos al DataFrame de noticias
for ticker in tickers:
    min_date = df_news[df_news['ticker'] == ticker]['date'].min().strftime('%Y-%m-%d')
    max_date = df_news[df_news['ticker'] == ticker]['date'].max().strftime('%Y-%m-%d')
    price_data = get_stock_prices(ticker, min_date, max_date)
    # Rellenar los precios faltantes para los días sin operaciones bursátiles
    price_data = price_data.reindex(pd.date_range(min_date, max_date), method='ffill')
    df_news.loc[df_news['ticker'] == ticker, 'stock_price'] = df_news[df_news['ticker'] == ticker]['date'].map(price_data)

# Visualización mejorada con Plotly
for ticker in tickers:
    filtered_df = df_news[df_news['ticker'] == ticker]
    # Crear un gráfico con dos ejes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Añadir gráfico de sentimiento
    fig.add_trace(
        go.Scatter(x=filtered_df['date'], y=filtered_df['compound'], name='Sentiment'),
        secondary_y=False,
    )

    # Añadir gráfico de precios de acciones
    fig.add_trace(
        go.Scatter(x=filtered_df['date'], y=filtered_df['stock_price'], name='Stock Price'),
        secondary_y=True,
    )

    # Añadir títulos y etiquetas
    fig.update_layout(
        title_text=f'{ticker} News Sentiment and Stock Price Over Time'
    )
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Sentiment', secondary_y=False)
    fig.update_yaxes(title_text='Stock Price', secondary_y=True)

    fig.show()

# Creación de tablas con información relevante
for ticker in tickers:
    filtered_df = df_news[df_news['ticker'] == ticker]
    print(f"Tabla para {ticker}")
    print(filtered_df[['date', 'title', 'compound', 'stock_price']])