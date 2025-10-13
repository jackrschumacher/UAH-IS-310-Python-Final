# Spam detector trainer

# Import the tools we need / if you have question about libraries ask me
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from nltk.tokenize import word_tokenize
import nltk 
nltk.download('punkt_tab')


# Load the data from a CSV file
# Make sure spam_data.csv is in the same folder as this script
# Convert the dataframes into lowercase in order to improve accuracy of the model
data = pd.read_csv(r"spam_data_large.csv")
data = data.apply(
    lambda col: col.astype(str).str.lower() if col.dtype == "object" else col
)
test_data = pd.read_csv(r"test_data.csv")
test_data = test_data.apply(
    lambda col: col.astype(str).str.lower() if col.dtype == "object" else col
)
print(data.columns)
# Tokenize the data in the dataframe in order to improve the accuracy of the model
data['tokenized_data'] = data["text"].apply(word_tokenize)
# Separate the message text (X) and the label (y)

X_train = data["text"]  # the actual message
y_train = data["label"]  # "spam" or "legitimate"

X_test = test_data["text"]
y_test = test_data["label"]

# Split the data into training and testing sets (80% train, 20% test)
# TODO: Trying out splitting files into two-one for testing and one for training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

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
