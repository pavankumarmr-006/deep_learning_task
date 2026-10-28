
import re
import numpy as np
import nltk
import gensim.downloader as api

from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. DOWNLOAD AND PREPARE THE DATASET
# --------------------------------------------------

nltk.download("movie_reviews", quiet=True)

documents = []
labels = []

# Use 500 positive and 500 negative reviews.
for category in ["pos", "neg"]:
    file_ids = movie_reviews.fileids(category)[:500]

    for file_id in file_ids:
        # Keep reviews short: use at most the first 40 words.
        text = movie_reviews.raw(file_id)
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        text = " ".join(tokens[:40])

        documents.append(text)
        labels.append(category)

print("Total texts:", len(documents))
print("Positive:", labels.count("pos"))
print("Negative:", labels.count("neg"))
print("Sample:", documents[0])

# --------------------------------------------------
# 2. TRAIN AND TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# --------------------------------------------------
# 3. TF-IDF + LOGISTIC REGRESSION
# --------------------------------------------------

tfidf_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]+\b",
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

tfidf_model.fit(X_train, y_train)
tfidf_predictions = tfidf_model.predict(X_test)

print("\n===== TF-IDF BASELINE =====")
print("Accuracy:",
      round(accuracy_score(y_test, tfidf_predictions), 4))
print("Macro F1:",
      round(f1_score(y_test, tfidf_predictions, average="macro"), 4))
print("\nClassification report:")
print(classification_report(y_test, tfidf_predictions))

# Confusion matrix
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(
    y_test, tfidf_predictions, labels=["neg", "pos"]
))

# --------------------------------------------------
# 4. TOP 10 IMPORTANT WORDS PER CLASS
# --------------------------------------------------

vectorizer = tfidf_model.named_steps["tfidf"]
classifier = tfidf_model.named_steps["classifier"]
feature_names = vectorizer.get_feature_names_out()

# For binary Logistic Regression, positive coefficients favor
# the class in classifier.classes_[1], and negative favor classes_[0].
coefficients = classifier.coef_[0]

print("\n===== TOP 10 IMPORTANT FEATURES =====")

for class_name, indices in [
    (classifier.classes_[0], np.argsort(coefficients)[:10]),
    (classifier.classes_[1], np.argsort(coefficients)[-10:][::-1])
]:
    print(f"\nClass: {class_name}")
    for index in indices:
        print(f"{feature_names[index]:20s} {coefficients[index]:.4f}")

# --------------------------------------------------
# 5. LOAD PRETRAINED GLOVE WORD VECTORS
# --------------------------------------------------

print("\nDownloading/loading GloVe vectors...")
glove = api.load("glove-wiki-gigaword-50")

# Find nearest words to two chosen words.
print("\n===== NEAREST WORDS =====")

for word in ["king", "computer"]:
    print(f"\n5 nearest words to '{word}':")
    print(glove.most_similar(word, topn=5))

# Word analogy: king - man + woman
print("\n===== WORD ANALOGY =====")
analogy = glove.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
print("king - man + woman ->", analogy[0][0])

# --------------------------------------------------
# 6. CREATE AVERAGED SENTENCE VECTORS
# --------------------------------------------------

def sentence_vector(text, model):
    words = re.findall(r"[a-z]+", text.lower())
    vectors = [model[word] for word in words if word in model]

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0)


X_train_vectors = np.vstack([
    sentence_vector(text, glove) for text in X_train
])
X_test_vectors = np.vstack([
    sentence_vector(text, glove) for text in X_test
])

# Standardize features and train Logistic Regression.
embedding_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

embedding_model.fit(X_train_vectors, y_train)
embedding_predictions = embedding_model.predict(X_test_vectors)

embedding_accuracy = accuracy_score(y_test, embedding_predictions)
embedding_f1 = f1_score(
    y_test, embedding_predictions, average="macro"
)

print("\n===== GLOVE SENTENCE-VECTOR RESULTS =====")
print("Accuracy:", round(embedding_accuracy, 4))
print("Macro F1:", round(embedding_f1, 4))
print("Confusion matrix:")
print(confusion_matrix(
    y_test, embedding_predictions, labels=["neg", "pos"]
))

# --------------------------------------------------
# 7. COMPARE BOTH APPROACHES
# --------------------------------------------------

print("\n===== MODEL COMPARISON =====")
print(f"{'Model':30s} {'Accuracy':>10s} {'Macro F1':>10s}")
print("-" * 52)
print(
    f"{'TF-IDF + Logistic Regression':30s} "
    f"{accuracy_score(y_test, tfidf_predictions):10.4f} "
    f"{f1_score(y_test, tfidf_predictions, average='macro'):10.4f}"
)
print(
    f"{'GloVe averages + Logistic Reg.':30s} "
    f"{embedding_accuracy:10.4f} "
    f"{embedding_f1:10.4f}"
)
