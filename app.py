import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="MindEmotion - AI Mental Health Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CUSTOM STYLING
# ==============================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .joy {background-color: #FFD700;}
    .sadness {background-color: #5DADE2;}
    .anger {background-color: #E74C3C;}
    .fear {background-color: #884EA0;}
    .love {background-color: #F1948A;}
    .surprise {background-color: #45B39D;}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL AND VECTORIZER
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("emotion_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ==============================
# HEADER SECTION
# ==============================
st.markdown('<h1 class="main-title">🧠 MindEmotion: AI for Mental Health</h1>', unsafe_allow_html=True)
st.markdown("""
### 💬 Emotional Well-being Classifier  
This tool uses **Natural Language Processing (NLP)** to detect emotions from text messages and raise mental health awareness.  
*Supporting UN SDG 3: Good Health & Well-Being*
""")

st.write("---")

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("📝 Emotion Check-In")
st.sidebar.markdown("Type your message or thought below:")
user_text = st.sidebar.text_area("How are you feeling today?", height=150)

st.sidebar.markdown("💡 Example: 'I'm feeling really down lately.'")
st.sidebar.write("---")

# ==============================
# MAIN PREDICTION AREA
# ==============================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Emotion Analysis Results")
    
    if st.sidebar.button("🔍 Analyze Emotion", type="primary"):
        if not user_text.strip():
            st.warning("Please enter a message to analyze.")
        else:
            # Transform and predict
            vector = vectorizer.transform([user_text])
            emotion = model.predict(vector)[0]
            
            # Display colored result card
            st.markdown(f'<div class="result-card {emotion}">Predicted Emotion: {emotion.upper()}</div>', unsafe_allow_html=True)
            
            st.success("✅ Analysis complete!")
            
            # Optional: provide interpretation
            explanations = {
                "joy": "You seem to be feeling positive and happy! Keep spreading the joy 😊",
                "sadness": "You might be feeling low. It’s okay to reach out for support 💙",
                "anger": "You seem upset. Try deep breathing or journaling to cool off 🔥",
                "fear": "You might be anxious or worried. Remember to take slow, deep breaths 🌿",
                "love": "You sound affectionate and warm-hearted ❤️",
                "surprise": "Something unexpected caught your attention! 😮"
            }
            
            st.info(explanations.get(emotion, "Stay mindful of your emotions and take care of your mental health."))
    
    st.write("---")
    st.subheader("📊 Emotion Distribution in Training Data")
    
    try:
        data = pd.read_csv("train.txt", sep=";", names=["Text", "Emotion"])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=data, x="Emotion", order=data["Emotion"].value_counts().index, palette="coolwarm", ax=ax)
        ax.set_title("Emotion Frequency in Dataset")
        st.pyplot(fig)
    except:
        st.info("Training dataset not found — chart unavailable.")

with col2:
    st.subheader("ℹ️ About This App")
    st.markdown("""
    **MindEmotion** applies **Machine Learning** to promote mental wellness through emotion detection.
    
    ### How It Works
    1. **Text Input:** You share a message or thought.
    2. **NLP Processing:** The model transforms text using TF-IDF.
    3. **ML Prediction:** A classifier predicts your dominant emotion.
    
    ### Why It Matters
    - Supports **SDG 3: Good Health & Well-being**
    - Promotes emotional self-awareness
    - Encourages early mental health support
    
    ### Model Overview
    - **Algorithm:** Logistic Regression (TF-IDF features)
    - **Accuracy:** 86.9%
    - **Dataset:** Kaggle “Emotions Dataset for NLP”
    """)
    
    st.markdown("#### 💚 Helpful Mental Health Resources")
    st.markdown("""
    - [BetterHelp](https://www.betterhelp.com) — Online therapy with professionals  
    - [Mind.org](https://www.mind.org.uk) — Mental health support and education  
    - [Talkspace](https://www.talkspace.com) — Digital therapy platform  
    - [Headspace](https://www.headspace.com) — Meditation & mindfulness
    """)

# ==============================
# FOOTER
# ==============================
st.write("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
💬 MindEmotion | Advancing SDG 3: Good Health & Well-being |
<b>Disclaimer:</b> This tool is for educational and awareness purposes only.  
If you're struggling, please reach out to a licensed mental health professional.
</div>
""", unsafe_allow_html=True)
