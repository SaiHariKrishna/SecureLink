# streamlit_app_cloud.py
"""
Streamlit Demo: URL Safety Classifier
- Automatically trains model on urls_500.csv
- Enter a URL to get Safe/Unsafe prediction
- Shows lexical features and probability
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="URL Safety Classifier", layout="centered")
st.title("URL Safety Classifier — Cloud Demo")
st.markdown("Enter a URL to see if it is safe or unsafe. This demo uses a Random Forest model trained on synthetic data.")

# --- 1. Load dataset ---
@st.cache_data
def load_data(path="urls_500.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

# --- 2. Feature extraction ---
def is_ip(hostname):
    return bool(re.match(r'^\d{1,3}(?:\.\d{1,3}){3}$', hostname))

def extract_features(url):
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    query = parsed.query or ''

    features = {}
    features['url_len'] = len(url)
    features['hostname_len'] = len(hostname)
    features['path_len'] = len(path)
    features['query_len'] = len(query)
    features['count_digits'] = sum(c.isdigit() for c in url)
    features['count_hyphen'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_dot'] = url.count('.')
    features['has_https'] = int(parsed.scheme == 'https')
    features['num_subdirs'] = path.count('/')
    features['has_ip'] = int(bool(re.search(r'//\d+\.\d+\.\d+\.\d+', url)) or is_ip(hostname))
    features['has_login_word'] = int(bool(re.search(r'login|signin|account|verify|update', url, re.I)))
    features['num_parts'] = len(hostname.split('.')) if hostname else 0
    features['has_at_symbol'] = int(features['count_at'] > 0)
    return features

def build_lexical_df(urls):
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows)

# --- 3. Train model (once) ---
@st.cache_resource
def train_model(df):
    urls = df['url'].astype(str).values
    labels = df['label'].astype(int).values
    lex_df = build_lexical_df(urls)
    tfv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=1500)
    X_tfidf = tfv.fit_transform(urls)
    X = np.hstack([lex_df.fillna(0).values, X_tfidf.toarray()])
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, labels)
    return clf, tfv, list(lex_df.columns)

clf, tfv, lex_cols = train_model(df)

# --- 4. Prediction function ---
def predict_url(url):
    lex_df = build_lexical_df([url])
    X_tfidf = tfv.transform([url])
    X = np.hstack([lex_df.fillna(0).values, X_tfidf.toarray()])
    label = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0,1]) if hasattr(clf, "predict_proba") else None
    return label, prob, lex_df.T

# --- 5. Streamlit UI ---
url_input = st.text_input("Enter URL here:", value="http://example.com/login")

if st.button("Predict"):
    if not url_input.strip():
        st.error("Please enter a URL")
    else:
        label, prob, features_df = predict_url(url_input)
        st.subheader("Prediction Result")
        st.write("**URL:**", url_input)
        st.write("**Classification:**", "⚠️ Unsafe" if label==1 else "✅ Safe")
        if prob is not None:
            st.write(f"**Probability (unsafe):** {prob:.3f}")
        st.subheader("Extracted Lexical Features")
        st.write(features_df)
        st.caption("Model: RandomForest trained on lexical + character n-gram TF-IDF features")
