# Email / SMS Spam Prediction

Car Price Prediction project jaisa hi structure follow karta hai ye project — EDA script, model training script, aur Streamlit app.

## Dataset
`email_spam_data.csv` — 5,572 labeled messages (`ham` / `spam`), classic SMS Spam Collection dataset (email spam classifiers is exact type ke dataset pe train hote hain).

## Project Structure
```
email_spam_project/
├── email_spam_data.csv        # raw dataset (Label, Message)
├── email_spam_eda.py          # Step 1: EDA + Cleaning + Text Preprocessing + TF-IDF + Train-Test Split
├── train_model.py             # Step 2: Model training (Naive Bayes, Logistic Regression, XGBoost) + best model save
├── spam_prediction.py         # Step 3: Streamlit app (final deployment)
├── requirements.txt
└── README.md
```

## Kaise Run Karein (conda environment mein)

```bash
# 1. Environment setup
conda create -n spam-predictor python=3.11 -y
conda activate spam-predictor
python -m pip install -r requirements.txt

# 2. EDA + preprocessing run karo (ye TF-IDF vectorizer aur train/test splits banayega)
python -m email_spam_eda

# 3. Model train karo (ye best model (spam_model.pkl) save karega)
python -m train_model

# 4. Streamlit app run karo
streamlit run spam_prediction.py
```

**Note:** `email_spam_eda.py` file `/mnt/user-data/uploads/email_spam_data.csv` path use kar raha hai (jaise original car price project mein tha). Apne local system pe run karte waqt is path ko `email_spam_data.csv` (same folder) se replace kar dena.

## Pipeline Summary

1. **EDA**: Class imbalance check (~87% ham, ~13% spam), message length analysis, top words per class.
2. **Cleaning**: Duplicates hataye, empty messages hataye.
3. **Text Preprocessing**: lowercase → remove URLs/emails/numbers/punctuation → remove stopwords → lemmatize.
4. **Feature Engineering**: TF-IDF vectorization (unigrams + bigrams, top 3000 features).
5. **Modeling**: Naive Bayes vs Logistic Regression vs XGBoost — F1-score se compare kiya (accuracy nahi, kyunki imbalanced data hai).
6. **Best model**: Naive Bayes (F1 ~0.89, spam precision 100%) — auto-saved as `spam_model.pkl`.
7. **Deployment**: Streamlit app text input leta hai aur real-time spam/ham prediction deta hai with confidence score.

## Key Design Decisions (jaise original project mein comments the)

- **Stratified split** use kiya train_test_split mein taaki spam/ham ratio train aur test dono mein preserve rahe.
- **TF-IDF fit sirf training data par** hua hai, test data pe sirf transform — taaki data leakage na ho (same principle jo car price project mein StandardScaler ke saath use hua tha).
- **F1-score** ko model selection metric banaya, accuracy nahi — imbalanced classification mein accuracy misleading hoti hai.
- **class_weight='balanced'** (Logistic Regression) aur **scale_pos_weight** (XGBoost) use kiya imbalance handle karne ke liye.
