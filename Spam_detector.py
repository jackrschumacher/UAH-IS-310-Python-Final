
# A script to predict if messages are spam or legitimate

# Import the tools we need
import pandas as pd
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model_trained.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Load the new messages you want to check
# Make sure this file has one column: "text"
messages = pd.read_csv("incoming_messages.csv")

# Convert the new messages into numbers using the same vectorizer
X_new = vectorizer.transform(messages["text"])

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
