import streamlit as st
import pickle
import re
import numpy as np

# --- Load model and vectorizer ---
model_p = pickle.load(open('svm_model.pkl', 'rb'))
vectorizer_p = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# --- Page configuration ---
st.set_page_config(page_title="🧠 Depression Detector", page_icon="💬")

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
    This app detects whether a tweet is expressing signs of **depression** or not using a trained **SVM model**.
    
    **Model:** Support Vector Machine (SVM)  
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

        # Optional: Get prediction confidence (works if model has `predict_proba`)
        try:
            confidence = model_p.predict_proba(vectorized)[0][prediction] * 100
        except:
            confidence = None

        # --- Result Display ---
        st.markdown("---")
        if prediction == 1:
            st.error("🧠 Depressed tweet detected")
        else:
            st.success("😊 Non-depressed tweet detected")

        if confidence:
            st.markdown(f"**Confidence:** {confidence:.2f}%")
