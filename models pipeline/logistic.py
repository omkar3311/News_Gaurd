import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from joblib import dump

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data_fake = pd.read_csv("Fake.csv")
data_true = pd.read_csv("True.csv")
data_fake["class"] = 0
data_true["class"] = 1
data = pd.concat([data_fake, data_true], axis=0)
data = data[['text', 'class']]
data['text'] = data['text'].apply(clean_text)
X = data['text']
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)

dump(model, "logistic+vectorization.joblib")
print("Model pipeline saved as logistic+vectorization.joblib")