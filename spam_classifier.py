import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and clean dataset
df = pd.read_csv("spam.csv", usecols=[0, 1], names=["label", "message"], skiprows=1, encoding='latin-1')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

# 2. Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

df['message'] = df['message'].apply(preprocess)

# 3. Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 4. TF-IDF Vectorization with tuning
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    max_df=0.95,
    min_df=5,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# 6. Evaluation
print("\n====== Naive Bayes Classifier ======")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Custom Message Test
test_msg = "click here now you will get cash of 10000 rupees to your accoun?"
test_msg_preprocessed = preprocess(test_msg)
test_vec = vectorizer.transform([test_msg_preprocessed])

spam_proba = model.predict_proba(test_vec)[0][1]
print(f"\nSpam probability: {spam_proba:.2f}")
print("Prediction for message:", test_msg)
print("Predicted as:", "Spam" if model.predict(test_vec)[0] else "Ham")
