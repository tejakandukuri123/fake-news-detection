import pandas as pd
# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine and shuffle datasets
data = pd.concat([fake, true])

data = data.sample(frac=1, random_state=42)

data.reset_index(drop=True, inplace=True)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

X = data["content"]

# Use label column as output
y = data["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)
X = vectorizer.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict using test data
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Take custom news input
news = input("\nEnter a news article:\n")

# Convert input text into numbers
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

# Show result
if prediction[0] == 0:
    print("\nPrediction: Fake News")
else:
    print("\nPrediction: Real News")

# Database section
import sqlite3

# Connect database
conn = sqlite3.connect("news_results.db")

# Create cursor
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    news TEXT,
    prediction TEXT
)
""")

# Store result
result_text = "Fake News" if prediction[0] == 0 else "Real News"

cursor.execute(
    "INSERT INTO results (news, prediction) VALUES (?, ?)",
    (news, result_text)
)

# Save changes
conn.commit()

# Close database
conn.close()

print("\nResult saved to database successfully.")

# Show saved records
conn = sqlite3.connect("news_results.db")

cursor = conn.cursor()

print("\nSaved Prediction History:\n")

for row in cursor.execute("SELECT * FROM results"):
    print(row)

conn.close()