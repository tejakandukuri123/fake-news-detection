import re
import webbrowser
import numpy as np
from difflib import SequenceMatcher
import tkinter as tk
from tkinter import ttk
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tkinter import messagebox
from datetime import datetime
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from news_analyzer import analyze_news

matched_link = ""

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
recent = pd.read_csv("recent_news.csv")

recent.columns = recent.columns.str.lower()

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true, recent])

data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str)

# Input and output

data["content"] = (
    data["title"].fillna("") +
    " " +
    data["text"].fillna("")
)

X = data["content"]

y = data["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

def refresh_news():

    os.system("python news_collector.py")

    messagebox.showinfo(
        "Success",
        "Latest news downloaded successfully!"
    )

# Prediction function
def detect_news():

    news = news_text.get("1.0", tk.END).strip()

    if not news:
        result_label.config(
            text="Please enter news text",
            fg="red"
        )
        return

    data = analyze_news(news, model, vectorizer)

    prediction = [data["prediction"]]
    result = data["result"]
    confidence = data["confidence"]

    color = "green" if prediction == 1 else "red"

    result_text = (
        f"Prediction: {result}\n"
        f"Confidence: {confidence:.2f}%"
    )

    score = data["score"]
    match = data["match"]
    link = data["link"]
    source_name = data["source_name"]

    common_words = data["common_words"]
    missing_words = data["missing_words"]

    keyword_score = data["keyword_score"]
    tfidf_score = data["tfidf_score"]

    top_matches = data["top_matches"]
    
    matched_keywords = ", ".join(common_words)
    top_matches_text = ""

    for i, item in enumerate(top_matches, 1):

        score_value = round(item[0] * 100, 1)

        title = item[1]

        publisher = item[2]

        top_matches_text += (
            f"{i}. {title}\n"
            f"   ({publisher}) - "
            f"{score_value}%\n\n"
        )

    missing_keywords = ", ".join(missing_words)

    print("Verification Score =", score)
    print("Top Match =", match)
   
    global matched_link
    matched_link = link

    print("Reached verification")
    print("Verification completed")
    print("Score =", score)
    print("Match =", match)

    print("Top Match =", match)

    if score > 0.15:

        open_button.config(state="normal")

        if prediction == 1:

            final_status = "REAL NEWS (ML + VERIFICATION)"
            final_color = "green"

        else:

            final_status = "REAL NEWS (VERIFIED BY TRUSTED SOURCE)"
            final_color = "darkgreen"

        if score >= 0.60:
            match_quality = "STRONG MATCH"

        elif score >= 0.30:
            match_quality = "PARTIAL MATCH"

        else:
            match_quality = "WEAK MATCH"

        verification_text = (
            f"\n\nVerification:\n"
            f"✓ Similar News Found\n"
            f"Verification Score: {score*100:.1f}%\n\n"
            f"Verification Method:\n"
            f"• Keyword Matching\n"
            f"• TF-IDF Vectorization\n"
            f"• Cosine Similarity\n\n"
            f"Keyword Score: {keyword_score*100:.1f}%\n"
            f"TF-IDF Score: {tfidf_score*100:.1f}%\n"
            f"Match Quality: {match_quality}\n\n"
            f"Matched Keywords:\n"
            f"{matched_keywords}\n\n"
            f"\nMissing Keywords:\n"
            f"{missing_keywords}\n"
            f"Top Match:\n"
            f"{match}\n\n"
            f"Top 3 Similar Articles:\n"
            f"{top_matches_text}\n"
            f"Source: {source_name}\n\n"
            f"Final Assessment:\n"
            f"{final_status}"
        )
           

    else:

        open_button.config(state="disabled")

        if prediction == 0:
            final_status = "FAKE NEWS (ML DETECTED)"

        else:
            final_status = "UNVERIFIED NEWS (NO MATCH FOUND)"

        verification_text = (
            f"\n\nVerification:\n"
            f"✗ No Similar News Found\n\n"
            f"Final Decision:\n"
            f"{final_status}"
        )

    print("Updating GUI")

    global report_news
    global report_result
    global report_confidence
    global report_score
    global report_match
    global report_status
    global report_source

    report_news = news
    report_result = result
    report_confidence = confidence
    report_score = score * 100
    report_match = match
    report_status = final_status
    report_source = source_name

    result_label.config(
        text=result_text + verification_text,
        fg=color
    )
    
    conn = sqlite3.connect("news_results.db")

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results(
        news TEXT,
        prediction TEXT,
        timestamp TEXT
    )
    """)

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    cursor.execute(
        "INSERT INTO results VALUES (?, ?, ?)",
        (news, result, current_time)
    )

    conn.commit()
    conn.close()

# Export history function
def export_history():

    conn = sqlite3.connect("news_results.db")

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM results")

    rows = cursor.fetchall()

    conn.close()

    # Create text file
    with open("history_export.txt", "w", encoding="utf-8") as file:

        for row in rows:
            file.write(f"News: {row[0]}\n")
            file.write(f"Prediction: {row[1]}\n")
            file.write("-" * 50 + "\n")

    result_label.config(
        text="History exported successfully"
    )

# Function to show history
def show_history():

    # Create new window
    history_window = tk.Toplevel(window)

    history_window.title("Prediction History")

    history_window.geometry("950x500")

    # Text area
    columns = ("Date & Time", "News", "Prediction")

    search_frame = tk.Frame(history_window)
    search_frame.pack(pady=5)

    tk.Label(search_frame, text="Search:").pack(side=tk.LEFT)

    search_entry = tk.Entry(search_frame, width=40)
    search_entry.pack(side=tk.LEFT, padx=5)

    tk.Button(
        search_frame,
        text="Search",
        command=lambda: load_history(search_entry.get())
    ).pack(side=tk.LEFT)

    tree = ttk.Treeview(
        history_window,
        columns=columns,
        show="headings",
        height=15
    )

    tree.heading("Date & Time", text="Date & Time")
    tree.heading("News", text="News")
    tree.heading("Prediction", text="Prediction")

    tree.column("Date & Time", width=170, anchor="center")
    tree.column("News", width=600)
    tree.column("Prediction", width=120, anchor="center")

    scrollbar = ttk.Scrollbar(
        history_window,
        orient="vertical",
        command=tree.yview
    )

    tree.configure(yscrollcommand=scrollbar.set)

    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def load_history(search_text=""):

        tree.delete(*tree.get_children())

        conn = sqlite3.connect("news_results.db")
        cursor = conn.cursor()

        if search_text:
            cursor.execute(
                """
                SELECT * FROM results
                WHERE news LIKE ?
                """,
                ('%' + search_text + '%',)
            )
        else:
            cursor.execute(
                "SELECT * FROM results"
            )

        rows = cursor.fetchall()
        print(rows)
        print("Total rows:", len(rows))

        conn.close()

        for row in rows:
            tree.insert(
                "",
                tk.END, 
                values=(
                    row[2],
                    row[0],
                    row[1]
                )
            )

    def load_selected_news(event):

        print("Double click detected")

        selected = tree.focus()
        print("Selected:", selected)

        if not selected:
            print("Nothing selected")
            return

        values = tree.item(selected)["values"]
        print(values)

        news = values[1]

        news_text.delete("1.0", tk.END)
        news_text.insert(tk.END, news)

        history_window.destroy()
    tree.bind("<Double-1>", load_selected_news)

    load_history()


def generate_report():

    report = result_label.cget("text")

    with open("verification_report.txt", "w",
              encoding="utf-8") as file:

        file.write(report)

    messagebox.showinfo(
        "Report",
        "verification_report.txt created"
    )

def open_article():
    global matched_link

    if matched_link:
        webbrowser.open(matched_link)

def generate_report():

    try:

        current_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        pdf = SimpleDocTemplate(
            "verification_report.pdf"
        )

        styles = getSampleStyleSheet()

        content = []

        content.append(
            Paragraph(
                "FAKE NEWS DETECTION AND VERIFICATION SYSTEM",
                styles["Title"]
            )
        )

        content.append(
            Paragraph(
                "Generated Analysis Report",
                styles["Heading2"]
            )
        )

        content.append(Spacer(1, 15))

        content.append(
            Paragraph(
                f"<b>Date & Time:</b> {current_time}",
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 15))

        content.append(
            Paragraph(
                "<b>NEWS ENTERED</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                report_news,
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 15))

        content.append(
            Paragraph(
                "<b>ANALYSIS RESULTS</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                f"<b>Prediction:</b> {report_result}",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                f"<b>Confidence:</b> {report_confidence:.2f}%",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                f"<b>Verification Score:</b> {report_score:.2f}%",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                f"<b>Matched Article:</b> {report_match}",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                f"<b>Source:</b> {report_source}",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                f"<b>Final Assessment:</b> {report_status}",
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 25))

        content.append(
            Paragraph(
                "<b>Developed By</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                "Kandukuri Sriteja",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                "Master of Computer Applications (MCA)",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                "Osmania University",
                styles["BodyText"]
            )
        )

        pdf.build(content)

        messagebox.showinfo(
            "Success",
            "Professional PDF Report Generated Successfully"
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )

# Create window
window = tk.Tk()

canvas = tk.Canvas(window)
scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)

scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def _on_mousewheel(event):
    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )

canvas.bind_all("<MouseWheel>", _on_mousewheel)

window.title("Fake News Detection System")
window.geometry("1200x1000")
window.configure(bg="#f5f5f5")

# Heading
title = tk.Label(
    scrollable_frame,
    text="Fake News Detection System",
    font=("Arial", 24, "bold"),
    bg="#f0f0f0",
    fg="darkblue"
)
title.pack(pady=20)

# Text area
news_text = tk.Text(scrollable_frame, height=10, width=70)
news_text.pack(pady=20)

# Result label
result_label = tk.Label(
    scrollable_frame,
    text="Prediction will appear here",
    font=("Arial", 18, "bold"),
    fg="blue",
    bg="#f0f0f0"
)

result_label.config(
    wraplength=850,
    justify="center"
)
result_label.pack(pady=10)

# detect Button
detect_button = tk.Button(
    scrollable_frame,
    text="Detect News",
    font=("Arial", 14, "bold"),
    bg="darkblue",
    fg="white",
    padx=20,
    pady=10,
    command=detect_news
)

detect_button.pack(pady=10)

#open button
open_button = tk.Button(
    scrollable_frame,
    text="Open Source Article",
    font=("Arial", 12, "bold"),
    bg="purple",
    fg="white",
    padx=20,
    pady=8,
    command=open_article
)

open_button.pack(pady=10)
open_button.config(state="disabled")

# History button
history_button = tk.Button(
    window,
    text="View History",
    font=("Arial", 12, "bold"),
    bg="green",
    fg="white",
    padx=20,
    pady=8,
    command=show_history
)

history_button.pack(pady=10)

# Clear function
def clear_text():
    news_text.delete("1.0", tk.END)
    result_label.config(text="Prediction will appear here")
    open_button.config(state="disabled")

# Clear button
clear_button = tk.Button(
    scrollable_frame,
    text="Clear",
    font=("Arial", 12, "bold"),
    bg="red",
    fg="white",
    padx=20,
    pady=8,
    command=clear_text
)

clear_button.pack(pady=10)

# Export button
export_button = tk.Button(
    window,
    text="Export History",
    font=("Arial", 12, "bold"),
    bg="purple",
    fg="white",
    padx=20,
    pady=8,
    command=export_history
)

export_button.pack(pady=10)

def show_statistics():

    conn = sqlite3.connect("news_results.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM results")
    total = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM results WHERE prediction='Real News'"
    )
    real_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM results WHERE prediction='Fake News'"
    )
    fake_count = cursor.fetchone()[0]

    conn.close()

    if total == 0:

        messagebox.showinfo(
            "Statistics",
            "No data available yet."
        )

        return

    real_percent = (real_count / total) * 100
    fake_percent = (fake_count / total) * 100

    stats_window = tk.Toplevel(window)

    stats_window.title("Statistics Dashboard")

    stats_window.geometry("600x500")

    stats_label = tk.Label(
        stats_window,
        text=(
            "FAKE NEWS DETECTION STATISTICS\n\n"

            f"Total Predictions : {total}\n\n"

            f"Real News : {real_count}\n"
            f"Fake News : {fake_count}\n\n"

            f"Real News Percentage : {real_percent:.1f}%\n"
            f"Fake News Percentage : {fake_percent:.1f}%\n\n"

            "Detection Pipeline:\n\n"

            "1. Passive Aggressive Classifier\n"
            "   → Performs initial Fake/Real news prediction.\n\n"

            "2. TF-IDF Vectorization\n"
            "   → Converts news text into numerical feature vectors.\n\n"

            "3. Keyword Extraction\n"
            "   → Identifies important words after removing stop words.\n\n"

            "4. Cosine Similarity Verification\n"
            "   → Compares the input news with trusted archived news.\n\n"

            "5. SQLite News Archive\n"
            "   → Stores verified news articles for comparison."
        ),
        font=("Arial", 12, "bold")
    )

    stats_label.pack(pady=20)

    plt.figure(figsize=(5, 5))

    plt.pie(
        [real_count, fake_count],
        labels=["Real News", "Fake News"],
        autopct="%1.1f%%"
    )

    plt.title("Real vs Fake News")

    plt.show()

stats_button = tk.Button(
    window,
    text="Show Statistics",
    font=("Arial", 12, "bold"),
    bg="orange",
    fg="white",
    padx=20,
    pady=8,
    command=show_statistics
)

refresh_button = tk.Button(
    window,
    text="Refresh Live News",
    font=("Arial", 12, "bold"),
    bg="cyan",
    fg="black",
    padx=20,
    pady=8,
    command=refresh_news
)

refresh_button.pack(pady=10)

report_button = tk.Button(
    scrollable_frame,
    text="Generate Report",
    font=("Arial", 12, "bold"),
    bg="darkgreen",
    fg="white",
    padx=20,
    pady=8,
    command=generate_report
)

report_button.pack(pady=10)

stats_button.pack(pady=10)

# Accuracy label
accuracy_label = tk.Label(
    window,
    text=f"Model Accuracy: {accuracy * 100:.2f}%",
    font=("Arial", 12, "bold"),
    bg="#f0f0f0",
    fg="darkgreen"
)

accuracy_label.pack(pady=10)

# Run GUI
window.mainloop()