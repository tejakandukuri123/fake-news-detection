import numpy as np
from news_collector import verify_news

def analyze_news(news, model, vectorizer):

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    confidence = float(
        np.max(model.predict_proba(news_vector)) * 100
    )

    result = "Real News" if prediction[0] == 1 else "Fake News"

    score, match, link, source_name, common_words, missing_words, keyword_score, tfidf_score, top_matches = verify_news(news)

    return {
        "prediction": prediction[0],
        "result": result,
        "confidence": confidence,
        "score": score,
        "match": match,
        "link": link,
        "source_name": source_name,
        "common_words": common_words,
        "missing_words": missing_words,
        "keyword_score": keyword_score,
        "tfidf_score": tfidf_score,
        "top_matches": top_matches
    }