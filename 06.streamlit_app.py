import streamlit as st
import pandas as pd
import joblib
import os
import re
import string

# --- Page Config ---
st.set_page_config(
    page_title="Disaster Tweets Classifier",
    page_icon="🚨",
    layout="wide"
)

# --- Helper Functions ---
def resolve_model_path(filename):
    possible_paths = [
        os.path.join("models", filename),
        os.path.join("src", filename),
        filename,
        os.path.join("..", "models", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def clean_text(text):
    """
    Basic text cleaning function to match notebook preprocessing.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers (optional, but often good for general sentiment)
    text = re.sub(r'\d+', '', text)
    
    return text

@st.cache_resource
def load_model():
    model_path = resolve_model_path("best_model.pkl")
    if model_path:
        return joblib.load(model_path)
    return None

# --- Main App ---
def main():
    st.title("🚨 Disaster Tweets Classifier")
    st.markdown("""
    **[TR]** Tweet'in gerçek bir felaket haberi olup olmadığını (1) veya olmadığını (0) tahmin eder.
    **[EN]** Predicts whether a tweet is about a real disaster (1) or not (0).
    """)

    model = load_model()

    if model is None:
        st.error("🚨 Model file (`best_model.pkl`) not found! Please run the notebook to train the model first.")
        return

    tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 Batch Prediction (CSV)"])

    # --- TAB 1: Manual Input ---
    with tab1:
        st.subheader("Analyze a Tweet")
        user_input = st.text_area("Enter Tweet Text", "There is a forest fire near my house!")
        
        if st.button("Predict"):
            if user_input:
                # Preprocess
                cleaned_text = clean_text(user_input)
                
                try:
                    # Predict (Model should be a pipeline taking text input)
                    prediction = model.predict([cleaned_text])[0]
                    proba = model.predict_proba([cleaned_text])[0][1] if hasattr(model, "predict_proba") else 0.0
                    
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("Disaster Probability", f"{proba:.2%}")
                    
                    if prediction == 1:
                        c2.error("Result: REAL DISASTER 🚨")
                    else:
                        c2.success("Result: Not a Disaster (Metaphor/Safe) ✅")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.info("Ensure the model pipeline includes the Vectorizer step.")

    # --- TAB 2: Batch Input ---
    with tab2:
        st.subheader("Upload CSV")
        st.info("Required column: `text`")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    st.write("Preview:", df.head())
                    
                    if st.button("Predict Batch"):
                        # Clean
                        df['clean_text'] = df['text'].apply(clean_text)
                        
                        # Predict
                        preds = model.predict(df['clean_text'])
                        df['target'] = preds
                        
                        st.success("Analysis Complete!")
                        st.write(df[['text', 'target']].head())
                        
                        # Download
                        # Create valid submission format: id, target
                        if 'id' in df.columns:
                            sub = df[['id', 'target']]
                        else:
                            sub = df[['target']]
                            
                        csv = sub.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions", csv, "submission.csv", "text/csv")
                        
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
