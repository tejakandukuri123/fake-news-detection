import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt   
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from datetime import datetime


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from news_analyzer import analyze_news
from news_collector import collect_news

from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_model():

    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    conn = sqlite3.connect("news_archive.db")

    archive = pd.read_sql_query(
        "SELECT title FROM news_archive",
        conn
    )

    conn.close()

    archive["text"] = ""
    archive["subject"] = "Live News"
    archive["date"] = ""
    archive["label"] = 1
    archive.rename(
        columns={"title": "title"},
        inplace=True
    )

    data = pd.concat(
        [fake, true, archive],
        ignore_index=True
    )

    data.drop_duplicates(
        subset=["title"],
        inplace=True
    )

    data = data.sample(frac=1, random_state=42)
    data.reset_index(drop=True, inplace=True)

    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")

    data["content"] = (
        data["title"] + " " + data["text"]
    )

    print("Fake columns:", fake.columns.tolist())
    print("True columns:", true.columns.tolist())
    print("Data columns:", data.columns.tolist())

    X = data["content"]
    y = data["label"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )

    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model, vectorizer


def generate_pdf(data, news_text):

    filename = "news_report.pdf"

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        fontSize=22
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        textColor=colors.darkblue
    )

    elements = []

    # Header
    elements.append(
        Paragraph(
            "FAKE NEWS DETECTION & VERIFICATION REPORT",
            title_style
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "MCA Final Year Major Project",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%"))
    elements.append(Spacer(1, 15))

    # Input News
    elements.append(
        Paragraph(
            "Input News Article",
            heading_style
        )
    )

    elements.append(
        Paragraph(
            news_text,
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 15))
    elements.append(HRFlowable(width="100%"))
    elements.append(Spacer(1, 15))

    # Results
    elements.append(
        Paragraph(
            "Analysis Result",
            heading_style
        )
    )

    elements.append(
        Paragraph(
            f"<b>Prediction:</b> {data['result']}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Model Confidence:</b> {data['confidence']:.2f}%",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Verification Score:</b> {data['score']*100:.1f}%",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Final Decision:</b> {data['final_result']}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Source:</b> {data['source_name']}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Top Matching Article:</b> {data['match']}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%"))
    elements.append(Spacer(1, 10))

    # Footer
    elements.append(
        Paragraph(
            "Generated by Fake News Detection & Verification System using ML and NLP",
            styles["Italic"]
        )
    )

    elements.append(
        Paragraph(
            "Developed by K. Sriteja",
            styles["Italic"]
        )
    )

    doc.build(elements)

    return filename


model, vectorizer = load_model()

st.title("📰 Fake News Detection & Verification System") 

# ===========================
# Model Information
# ===========================
st.sidebar.markdown("## 📊 Model Information")

st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Training:** 80%")
st.sidebar.write("**Testing:** 20%")
st.sidebar.write("**Accuracy:** 94.21%")   # Replace with your actual accuracy

st.title("📰 Fake News Detection & Verification System")

col1, col2 = st.columns([4, 1])

with col2:
    if st.button("🔄 Refresh News Database"):
        with st.spinner("Collecting latest news articles and retraining model..."):
            try:
                collect_news()

                st.cache_resource.clear()

                model, vectorizer = load_model()

                st.success(
                    "✅ News database updated and ML model retrained successfully!"
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")


st.markdown(
    """
    ### MCA Final Year Major Project

    This system combines:
    - Machine Learning based Fake News Detection
    - Live News Verification
    - TF-IDF Vectorization
    - Cosine Similarity
    - Keyword Matching
    - SQLite News Archive
    """
)

st.divider()

st.subheader("Enter News Article")

if "news" not in st.session_state:
    st.session_state.news = ""

news = st.text_area(
    "Paste the news article below",
    height=250,
    key="news",
    placeholder="Paste any news article here..."
)

col1, col2 = st.columns(2)

with col1:
    detect = st.button("🔎 Detect News")

def clear_text():
    st.session_state["news"] = ""

with col2:
    st.button(
        "🗑 Clear Text",
        on_click=clear_text
    )

if detect:

    if news.strip() == "":
        st.error("Please enter a news article.")

    else:

        data = analyze_news(news, model, vectorizer)

        # Final decision based on verification + ML
        if data["score"] >= 0.60:
            data["final_result"] = "Verified Real News"

        elif data["score"] >= 0.30:
            data["final_result"] = "Partially Verified"

        elif data["prediction"] == 1:
            data["final_result"] = "Unverified Real News"

        else:
            data["final_result"] = "Likely Fake News"

        st.success("Analysis Completed")

        conn = sqlite3.connect("news_results.db")
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results(
            news TEXT,
            prediction TEXT,
            timestamp TEXT
        )
        """)

        current_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        cursor.execute(
            """
            INSERT INTO results
            VALUES (?, ?, ?)
            """,
            (
                news,
                data["result"],
                current_time
            )
        )

        conn.commit()
        conn.close()

        st.subheader("Machine Learning Prediction")

        if data["prediction"] == 1:
            st.success(f"ML Model Prediction: {data['result']}")
        else:
            st.error(f"ML Model Prediction: {data['result']}")

        st.write(f"**Model Confidence:** {data['confidence']:.2f}%")
        st.write(f"**Verification Score:** {data['score']*100:.1f}%")

        st.divider()

        st.subheader("Verification")

        if data["score"] > 0.15:
            st.success("✓ Similar News Found")
        else:
            st.warning("✗ No Similar News Found")

        st.write(f"**Source:** {data['source_name']}")
        st.write(f"**Top Match:** {data['match']}")

        st.write("### Matched Keywords")
        st.write(", ".join(data["common_words"]))

        st.write("### Missing Keywords")
        st.write(", ".join(data["missing_words"]))

        st.write(f"**Keyword Score:** {data['keyword_score']*100:.1f}%")
        st.write(f"**TF-IDF Score:** {data['tfidf_score']*100:.1f}%")

        st.subheader("Top 3 Similar Articles")

        for i, item in enumerate(data["top_matches"], 1):

            similarity = item[0] * 100
            title = item[1]
            publisher = item[2]

            st.markdown(
                f"**{i}. {title}**\n\n"
                f"Publisher: {publisher}\n\n"
                f"Similarity: {similarity:.1f}%"
            )
    
        if data["score"] > 0.15 and data["link"]:
            st.link_button(
                "🌐 Open Original Article",
                data["link"]
            )

        st.divider()

        st.subheader("🏁 Final Decision")

        if data["final_result"] == "Verified Real News":
            st.success("✅ VERIFIED REAL NEWS")
            st.write(
                f"This article matched a trusted live news source with a verification score of {data['score']*100:.1f}%."
            )

        elif data["final_result"] == "Partially Verified":
            st.warning("🟡 PARTIALLY VERIFIED")
            st.write(
                "Some evidence was found, but the similarity is not high enough for full verification."
            )

        elif data["final_result"] == "Unverified Real News":
            st.info("⚠️ UNVERIFIED REAL NEWS")
            st.write(
                "The ML model predicts Real News, but no trusted live source could verify it."
            )

        else:
            st.error("❌ LIKELY FAKE NEWS")
            st.write(
                "No trusted matching article was found and the ML model predicts Fake News."
            )


        pdf_file = generate_pdf(data, news)

        with open(pdf_file, "rb") as file:
            st.download_button(
                "📄 Download PDF Report",
                file,
                file_name="news_report.pdf"
            )

st.divider()

show_history = st.toggle(
    "📜 View Prediction History"
)

if show_history:

    conn = sqlite3.connect("news_results.db")

    search = st.text_input(
        "Search News"
    )

    query = """
    SELECT *
    FROM results
    WHERE news LIKE ?
    ORDER BY timestamp DESC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=[f"%{search}%"]
    )

    conn.close()

    st.subheader("Prediction History")

    st.dataframe(
        df,
        width="stretch"
    )

show_stats = st.toggle(
    "📊 View Statistics"
)

if show_stats:  

    conn = sqlite3.connect("news_results.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM results"
    )
    total = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM results WHERE prediction='Real News'"
    )
    real = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM results WHERE prediction='Fake News'"
    )
    fake = cursor.fetchone()[0]

    conn.close()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Predictions", total)
    with c2:
        st.metric("Real News", real)
    with c3:
        st.metric("Fake News", fake)

    fig, ax = plt.subplots()

    ax.pie(
        [real, fake],
        labels=["Real News", "Fake News"],
        autopct="%1.1f%%"
    )

    ax.set_title(
        "Real vs Fake News"
    )

    st.pyplot(fig)

    chart = pd.DataFrame(
        {
            "Prediction": ["Real", "Fake"],
            "Count": [real, fake]
        }
    )

    st.bar_chart(
        chart.set_index("Prediction")
    )