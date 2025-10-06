
# Spam detector trainer

# Import the tools we need / if you have question about libraries ask me
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the data from a CSV file
# Make sure spam_data.csv is in the same folder as this script
data = pd.read_csv("spam_data.csv")

# Separate the message text (X) and the label (y)
X = data["text"]          # the actual message
y = data["label"]         # "spam" or "legitimate"

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Turn the text into numbers using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Test the model and show the accuracy
accuracy = model.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)

# Save the trained model and vectorizer so we can use them later
joblib.dump(model, "spam_model_trained.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("Model and vectorizer saved!")
