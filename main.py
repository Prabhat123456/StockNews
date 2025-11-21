import os
from fastapi import FastAPI, HTTPException
from gnews import GNews
from newspaper import Article, Config
import google.generativeai as genai

import datetime

# 1. CONFIGURATION
# ---------------------------
# PASTE YOUR GEMINI KEY HERE or set as environment variable

genai.configure(api_key="AIzaSyBgHuerfCXVdnWC9uXU1ZIQKAii9QJduS8")

model = genai.GenerativeModel("gemini-1.5-flash")

# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBgHuerfCXVdnWC9uXU1ZIQKAii9QJduS8")

app = FastAPI(title="AI Stock Analyst Agent")

# Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Scraper (to look like a real browser)
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10  # Don't wait too long for slow sites


# 2. HELPER: SCRAPE & ANALYZE
# ---------------------------
def analyze_article(stock_ticker, url, title):
    try:
        # A. Scrape the article content
        article = Article(url, config=config)
        article.download()
        article.parse()
        content = article.text[:4000]  # Limit text to save tokens/speed

        # Fallback if scraping fails/is empty
        if len(content) < 50:
            content = f"Could not scrape full text. Base analysis on headline: {title}"

        # B. Ask the LLM (Gemini)
        prompt = f"""
        You are a financial analyst. Analyze this news regarding {stock_ticker}.
        Article Title: {title}
        Article Content: {content}

        Return a strictly formatted response with these 3 lines:
        1. SUMMARY: (One concise sentence summarizing the event)
        2. IMPACT: (Choose one: Bullish, Bearish, or Neutral)
        3. REASON: (One sentence explaining WHY this affects the stock price)
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"SUMMARY: Error analyzing.\nIMPACT: Neutral\nREASON: {str(e)}"


# 3. API ENDPOINT
# ---------------------------
@app.get("/agent/smart-news")
def get_smart_news(ticker: str, limit: int = 3):
    """
    Fetches news, scrapes content, and uses LLM to analyze impact.
    Limit is set to 3 by default to ensure speed (LLMs take time!).
    """
    print(f"🔍 Analyzing {ticker}...")

    # A. Fetch Links
    google_news = GNews(language='en', country='IN', period='2d', max_results=limit)
    news_results = google_news.get_news(f"{ticker} stock news India")

    analyzed_data = []

    # B. Process Each Article
    for news in news_results:
        # Parse the LLM response
        raw_analysis = analyze_article(ticker, news['url'], news['title'])

        # Simple string parsing to make JSON pretty
        # (In production, you might want the LLM to return pure JSON)
        lines = raw_analysis.split('\n')
        summary = next((line.split(': ', 1)[1] for line in lines if 'SUMMARY' in line), "No summary")
        impact = next((line.split(': ', 1)[1] for line in lines if 'IMPACT' in line), "Neutral")
        reason = next((line.split(': ', 1)[1] for line in lines if 'REASON' in line), "No reason provided")

        analyzed_data.append({
            "title": news['title'],
            "link": news['url'],
            "published": news['published date'],
            "ai_summary": summary,
            "market_impact": impact,
            "impact_reason": reason
        })

    return {
        "stock": ticker,
        "analysis_engine": "Gemini 1.5 Flash",
        "articles": analyzed_data
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)