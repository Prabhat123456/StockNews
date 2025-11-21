from fastapi import FastAPI, HTTPException
from gnews import GNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import datetime

# 1. Initialize App and Downloads
app = FastAPI(title="Indian Stock News & Sentiment Agent")

# Download VADER lexicon (run once)
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()


# 2. Helper Function: Analyze Sentiment
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    # Classify sentiment based on compound score
    if compound >= 0.05:
        return "Positive", compound
    elif compound <= -0.05:
        return "Negative", compound
    else:
        return "Neutral", compound


# 3. The Agent API Endpoint
@app.get("/agent/stock-news")
def get_stock_news(ticker: str, limit: int = 10):
    """
    Fetches news for a given Indian stock and returns sentiment.
    Example Ticker: 'Tata Motors', 'Reliance Industries', 'HDFC Bank'
    """
    try:
        # Configure Google News for India
        google_news = GNews(language='en', country='IN', period='2d', max_results=limit)

        # Search query: Combine ticker with 'Stock' to ensure relevance
        search_query = f"{ticker} stock news India"
        news_results = google_news.get_news(search_query)

        analyzed_news = []
        total_sentiment_score = 0
        article_count = 0

        for article in news_results:
            title = article.get('title', '')
            link = article.get('url', '')
            date = article.get('published date', '')

            # Analyze sentiment of the headline
            sentiment_label, sentiment_score = get_sentiment(title)

            analyzed_news.append({
                "title": title,
                "link": link,
                "published_at": date,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score
            })

            total_sentiment_score += sentiment_score
            article_count += 1

        # Calculate overall average sentiment
        overall_sentiment = "Neutral"
        if article_count > 0:
            avg_score = total_sentiment_score / article_count
            if avg_score >= 0.05:
                overall_sentiment = "Bullish (Positive)"
            elif avg_score <= -0.05:
                overall_sentiment = "Bearish (Negative)"

        return {
            "stock": ticker,
            "generated_at": datetime.datetime.now().isoformat(),
            "overall_sentiment": overall_sentiment,
            "news_count": article_count,
            "articles": analyzed_news
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 4. Run locally (optional)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)