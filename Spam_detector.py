# A script to predict if messages are spam or legitimate

# Import the tools we need
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Initialize stop_words variable and the lemmatizer
# Stop words are words that are non-useful for text analysis (or, and,etc)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Create a tokenizer to help process the text further
def text_tokenizer(text):
    # Only allow certain character
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert text into lowercase
    text = text.lower()
    # Tokenize the text
    tokenized_text = word_tokenize(text)
    # Create a list of cleaned tokens if the lenght of the token is greater than 2. If token is not in the stop words list, append to the tokens list
    cleaned_tokens = []
    for token in tokenized_text:
        if len(token) < 2:
            continue
        if token not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
    return cleaned_tokens


# Load the trained model and vectorizer
model = joblib.load("spam_model_trained.joblib")
vectorizer = joblib.load("vectorizer.joblib")


# Load the new messages you want to check
# Make sure this file has one column: "text"
# Create a copy of the messages dataframe and convert to lowercase to improve detection
messages = pd.read_csv("incoming_messages.csv")
messages_cleaned = messages.copy(deep=True)
messages_cleaned = messages_cleaned.apply(
    lambda col: col.astype(str).str.lower() if col.dtype == "object" else col
)

# Convert the new messages into numbers using the same vectorizer
X_new = vectorizer.transform(messages_cleaned["text"])

# Use the model to predict spam or legitimate
predictions = model.predict(X_new)

# Add the predictions to the messages
messages["prediction"] = predictions

# Save the results to a text file
with open("results.txt", "w") as f:
    for i, row in messages.iterrows():
        f.write(f"Message: {row['text']}\n")
        f.write(f"Prediction: {row['prediction']}\n")
        f.write("-" * 40 + "\n")

print("Predictions saved to results.txt")
