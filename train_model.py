# train_model.py

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import build_lexical_df, build_tfidf, combine_features

DEFAULT_MODEL_DIR = "models"
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'url' not in df.columns:
        raise ValueError("CSV must contain a 'url' column.")
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0 safe, 1 unsafe).")
    df = df[['url', 'label']].dropna().reset_index(drop=True)
    return df

def main(args):
    df = load_data(args.csv)
    urls = df['url'].astype(str).values
    labels = df['label'].astype(int).values

    # lexical features
    lex_df = build_lexical_df(urls)

    # TF-IDF features
    X_tfidf, tfv = build_tfidf(urls, n_features=args.tfidf_features, ngram_range=(3,5))

    # combine
    X = combine_features(lex_df, X_tfidf)
    y = labels

    # train/test split
    X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
        X, y, urls, test_size=args.test_size, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )

    print("Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer and lexical column names
    artifacts = {
        'clf': clf,
        'tfv': tfv,
        'lex_cols': list(lex_df.columns)
    }
    model_path = os.path.join(DEFAULT_MODEL_DIR, "phish_rf.joblib")
    joblib.dump(artifacts, model_path)
    print(f"Saved model artifacts to {model_path}")

    # Save a small test sample for demo
    sample_out = os.path.join(DEFAULT_MODEL_DIR, "test_sample_predictions.csv")
    out_df = pd.DataFrame({
        'url': urls_test,
        'true_label': y_test,
        'pred_label': y_pred,
        'pred_prob': (y_prob.tolist() if y_prob is not None else [None]*len(y_pred))
    })
    out_df.to_csv(sample_out, index=False)
    print(f"Wrote test predictions to {sample_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV with columns 'url' and 'label'")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--tfidf-features", type=int, default=1500)
    args = p.parse_args()
    main(args)
