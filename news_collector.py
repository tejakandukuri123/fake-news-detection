import feedparser
import pandas as pd
import sqlite3
from datetime import datetime

def collect_news():
    feeds = [
        "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.reutersagency.com/feed/?best-topics=world&post_type=best"

        "https://news.google.com/rss/headlines/section/topic/NATION?hl=en-IN&gl=IN&ceid=IN:en"
        "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en"
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en"
        "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en" 
        "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-IN&gl=IN&ceid=IN:en"

         # Indian National
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "https://indianexpress.com/feed/",
        "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
        "https://www.ndtv.com/rss",

        # Business
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://economictimes.indiatimes.com/rssfeedsdefault.cms",

        # State and Regional
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://www.thehindu.com/news/national/telangana/feeder/default.rss",
        "https://www.thehindu.com/news/national/andhra-pradesh/feeder/default.rss",

        # Technology
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://www.thehindu.com/sci-tech/technology/feeder/default.rss",

        # International
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.theguardian.com/world/rss"
    ]

    all_news = []

    for url in feeds:

        feed = feedparser.parse(url)

        for entry in feed.entries:

            publisher = "Unknown"

            if "-" in entry.title:
                publisher = entry.title.split("-")[-1].strip()

            all_news.append({
                "title": entry.title,
                "link": entry.link,
                "publisher": publisher
            })

    df = pd.DataFrame(all_news)

    df.to_csv("live_news.csv", index=False)

    conn = sqlite3.connect("news_archive.db")

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_archive(
        title TEXT UNIQUE,
        link TEXT,
        publisher TEXT,
        date_added TEXT
    )
    """)
    for _, row in df.iterrows():

        try:

            cursor.execute(
                """
                INSERT INTO news_archive
                (title, link, publisher, date_added)
                VALUES (?, ?, ?, ?)
                """,
                (
                    row["title"],
                    row["link"],
                    row["publisher"],
                    datetime.now().strftime("%Y-%m-%d")
                )
            )

        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()

    print(f"Collected {len(df)} news articles")

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_keywords(text):

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "in", "on", "at", "of", "to", "for", "with",
        "and", "or", "by", "from", "that", "this",
        "it", "as", "be", "have", "has", "had",
        "will", "would", "could", "should", "do",
        "does", "did", "if", "but", "not",

        "said", "says", "say",
        "new", "latest",
        "today", "yesterday",
        "report", "reports",
        "after", "before",
        "over", "under",
        "into", "about",
        "against", "during",
        "through", "while",
        "their", "there",
        "they", "them"
    }

    words = re.findall(r"\w+", text.lower())

    keywords = [
        word
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return set(keywords)

def verify_news(news_text):

    try:

        conn = sqlite3.connect("news_archive.db")

        archive_news = pd.read_sql_query(
            "SELECT * FROM news_archive",
            conn
        )

        conn.close()

        best_match = ""
        best_link = ""
        best_publisher = ""

        best_common_words = set()

        highest_score = 0
        top_matches = []

        best_keyword_score = 0
        best_tfidf_score = 0

        input_keywords = extract_keywords(news_text)

        all_titles = archive_news["title"].astype(str).tolist()

        documents = [news_text] + all_titles

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(
            documents
        )

        max_possible_score = sum(
            len(word)
            for word in input_keywords
        )

        print("Input Keywords:", input_keywords)
        print("Max Possible Score:", max_possible_score)

        for index, row in archive_news.iterrows():

            title = str(row["title"])

            tfidf_score = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[index + 1:index + 2]
            )[0][0]

            title_keywords = extract_keywords(title)

            common_words = input_keywords.intersection(
                title_keywords
            )

            keyword_score = sum(
                len(word)
                for word in common_words
            )

            if max_possible_score > 0:
                keyword_score = (
                    keyword_score / max_possible_score
                )

            final_score = (
                0.7 * keyword_score +
                0.3 * tfidf_score
            )

            if final_score >= 0.15:
                top_matches.append(
                    (
                        final_score,
                        title,
                        row["publisher"]
                    )
                )
            
            print("Keywords:", round(keyword_score * 100, 1))
            print("TFIDF:", round(tfidf_score * 100, 1))
            print("Final:", round(final_score * 100, 1))

            if final_score > highest_score:

                highest_score = final_score

                best_keyword_score = keyword_score
                best_tfidf_score = tfidf_score

                best_common_words = common_words

                best_match = title

                best_link = row["link"]

                best_publisher = row["publisher"]  

        missing_words = input_keywords - best_common_words
        
        top_matches.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        top_matches = top_matches[:3]

        return (
            highest_score,
            best_match,
            best_link,
            best_publisher,
            best_common_words,
            missing_words,
            best_keyword_score,
            best_tfidf_score,
            top_matches
        )

    except Exception as e:

        print("Verification Error:", e)

        return (
            0,
            "No verification data found",
            "",
            "",
            set(),
            set(),
            0,
            0,
            []
        )
    

if __name__ == "__main__":
    collect_news()