import os
import re
import pickle
import numpy as np
import streamlit as st

# --- Page configuration ---
st.set_page_config(page_title="🧠 Depression Detector", page_icon="💬")

# --- Debug: Show current directory and files ---
# Remove or comment out in production
st.write("Working directory:", os.getcwd())
st.write("Files in working dir:", os.listdir())

# --- Load model and vectorizer using relative paths ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'logistic_model.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')

model_p = pickle.load(open(model_path, 'rb'))
vectorizer_p = pickle.load(open(vectorizer_path, 'rb'))

# --- Clean function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'#\w+|@\w+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# --- Sidebar ---
with st.sidebar:
    st.title("ℹ️ About the App")
    st.write("""
    This app detects whether a tweet is expressing signs of **depression** using a trained ML model.
    
    **Model:** Logistic Regression  
    **Text Vectorization:** TF-IDF  
    **SMOTE Applied:** Yes  
    **Accuracy:** ~77.5%
    """)

# --- Main interface ---
st.title("💬 Tweets Depression Detection App")
st.markdown("Enter a tweet and the app will predict if it's expressing **depression**.")

# --- User Input ---
user_input = st.text_area("📝 Write your tweet here:")

# --- Prediction ---
if st.button('🔍 Predict'):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer_p.transform([cleaned])
        prediction = model_p.predict(vectorized.toarray())[0]

        # Optional: Get prediction confidence (if model supports it)
        try:
            confidence = model_p.predict_proba(vectorized)[0][prediction] * 100
        except AttributeError:
            confidence = None

        # --- Result Display ---
        st.markdown("---")
        if prediction == 1:
            st.error("🧠 Depressed tweet detected")
        else:
            st.success("😊 Non-depressed tweet detected")

        if confidence:
            st.markdown(f"**Confidence:** {confidence:.2f}%")

