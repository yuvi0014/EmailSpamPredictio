import re
import string
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)


def main():
    html_temp = """<h1> Email / SMS Spam Prediction</h1>"""
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("spam_model.pkl")   # Naive Bayes ya Logistic Regression ke liye
    # Agar best model XGBoost nikla ho to is line ko replace karo:
    # import xgboost as xgb
    # model = xgb.XGBClassifier()
    # model.load_model("spam_model.json")

    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("this app will help you predict whether a message is Spam or Ham (not spam).")

    message = st.text_area("Enter the email / SMS text:", height=150,
                            placeholder="e.g. Congratulations! You have won a free prize. Click here to claim now...")

    if st.button("Predict"):
        try:
            if not message.strip():
                st.warning("Please enter some text first.")
            else:
                cleaned = clean_text(message)
                vectorized = tfidf.transform([cleaned])
                pred = model.predict(vectorized)[0]

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vectorized)[0]
                    confidence = proba[pred] * 100
                else:
                    confidence = None

                if pred == 1:
                    if confidence:
                        st.error(f"🚨 This message is SPAM ({confidence:.1f}% confidence)")
                    else:
                        st.error("🚨 This message is SPAM")
                else:
                    if confidence:
                        st.success(f"✅ This message is HAM / Not Spam ({confidence:.1f}% confidence)")
                    else:
                        st.success("✅ This message is HAM / Not Spam")
        except Exception as e:
            st.error("Something went wrong while predicting: {}".format(e))


if __name__ == '__main__':
    main()
