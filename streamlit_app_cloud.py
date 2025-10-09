# ---------------------------
# streamlit_app_cloud.py
# ---------------------------

import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Helper functions
# ---------------------------

# --- URL parsing / lexical helpers ---
def get_registered_domain(hostname: str) -> str:
    if not hostname:
        return ""
    parts = hostname.lower().split('.')
    if len(parts) < 2:
        return hostname.lower()
    if len(parts) >= 3 and len(parts[-2]) <= 3 and len(parts[-1]) <= 3:
        return '.'.join(parts[-2:])
    return '.'.join(parts[-2:])

def get_tld(hostname: str) -> str:
    if not hostname or '.' not in hostname:
        return ''
    return hostname.lower().split('.')[-1]

def edit_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        cur = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, start=1):
            insert_cost = cur[j-1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j-1] + (0 if c1 == c2 else 1)
            cur[j] = min(insert_cost, delete_cost, replace_cost)
        prev = cur
    return prev[-1]

# ---------------------------
# Rule-based / whitelist
# ---------------------------
TRUSTED_DOMAINS = {
    "youtube.com", "google.com", "facebook.com", "twitter.com", "linkedin.com",
    "github.com", "amazon.com", "netflix.com", "wikipedia.org", "apple.com",
    "microsoft.com", "outlook.com", "gmail.com", "reddit.com", "stackoverflow.com"
}

SUSPICIOUS_TLD_PATTERNS = {"comx", "con", "cm", "c0m", "coom", "comm", "xn--"}
MAX_COMMON_TLD_LEN = 6
COMMON_TLDS = {"com","org","net","edu","gov","io","co","info","biz","in","uk","us","me"}
TYPOSQUAT_DISTANCE = 1
UNSAFE_PROB_THRESHOLD = 0.70

def rule_based_check(url: str):
    u = url.strip()
    parsed = urlparse(u if "://" in u else "http://" + u)
    hostname = (parsed.hostname or "").lower()
    reg_domain = get_registered_domain(hostname)
    tld = get_tld(hostname)

    # Whitelist check
    if reg_domain in TRUSTED_DOMAINS:
        return "whitelist_safe", f"Registered domain '{reg_domain}' is in trusted whitelist."

    # Suspicious TLD / malformed ending
    if tld in SUSPICIOUS_TLD_PATTERNS:
        return "suspicious_tld", f"TLD '{tld}' matches suspicious pattern."
    if len(tld) > MAX_COMMON_TLD_LEN and tld not in COMMON_TLDS:
        return "suspicious_tld", f"TLD '{tld}' is unusually long or unknown."

    # Typo-squatting check
    left = reg_domain.split('.')[0]
    for trusted in TRUSTED_DOMAINS:
        trusted_left = trusted.split('.')[0]
        if trusted_left == left:
            continue
        dist = edit_distance(left, trusted_left)
        if dist <= TYPOSQUAT_DISTANCE:
            return "typosquat", f"Domain '{reg_domain}' is edit-distance {dist} from trusted '{trusted}'."

    return None, "No rule-fired; use ML model."

# ---------------------------
# Lexical feature builder
# ---------------------------
def build_lexical_df(urls):
    data = []
    for url in urls:
        u = url.strip()
        parsed = urlparse(u if "://" in u else "http://" + u)
        hostname = parsed.hostname or ""
        url_len = len(u)
        num_dots = u.count('.')
        has_https = 1 if u.startswith('https://') else 0
        tld_len = len(get_tld(hostname))
        data.append([url_len, num_dots, has_https, tld_len])
    df = pd.DataFrame(data, columns=['url_len', 'num_dots', 'has_https', 'tld_len'])
    return df

# ---------------------------
# Load dataset and train ML model
# ---------------------------
@st.cache_data
def train_model():
    df = pd.read_csv("urls_500.csv")  # your CSV
    X_lex = build_lexical_df(df['url'])
    tfv = TfidfVectorizer()
    X_tfidf = tfv.fit_transform(df['url'])
    X = np.hstack([X_lex.fillna(0).values, X_tfidf.toarray()])
    y = df['label'].values
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf, tfv

clf, tfv = train_model()

# ---------------------------
# Prediction with rules + ML
# ---------------------------
def predict_with_rules(url, clf, tfv, lex_build_fn):
    rule_decision, rule_reason = rule_based_check(url)
    if rule_decision == "whitelist_safe":
        return {"final_label": 0, "prob": 0.01, "reason": rule_reason}
    if rule_decision == "suspicious_tld":
        return {"final_label": 1, "prob": 0.99, "reason": rule_reason}
    if rule_decision == "typosquat":
        return {"final_label": 1, "prob": 0.95, "reason": rule_reason}

    lex_df = lex_build_fn([url])
    X_tfidf = tfv.transform([url])
    X = np.hstack([lex_df.fillna(0).values, X_tfidf.toarray()])
    prob = float(clf.predict_proba(X)[0,1]) if hasattr(clf, "predict_proba") else 0.0
    if prob >= UNSAFE_PROB_THRESHOLD:
        return {"final_label": 1, "prob": prob, "reason": f"ML prob >= {UNSAFE_PROB_THRESHOLD}"}
    else:
        return {"final_label": 0, "prob": prob, "reason": f"ML prob < {UNSAFE_PROB_THRESHOLD}"}

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("SecureLink - URL Safety Checker")
st.write("Enter a URL to check if it's safe or unsafe:")

url_input = st.text_input("URL", "https://youtube.com")

if st.button("Check URL"):
    result = predict_with_rules(url_input, clf, tfv, build_lexical_df)
    label = result['final_label']
    prob = result['prob']
    reason = result['reason']

    st.subheader("Prediction Result")
    st.write("**URL:**", url_input)
    st.write("**Classification:**", "⚠️ Unsafe" if label==1 else "✅ Safe")
    st.write(f"**Probability (unsafe):** {prob:.3f}")
    st.write("**Reason:**", reason)
