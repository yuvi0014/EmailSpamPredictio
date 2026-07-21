Here is the complete English translation of your text, maintaining the original structure, formatting, and technical depth:

---

# Email / SMS Spam Prediction

This project follows the same structure as the Car Price Prediction project — featuring an EDA script, a model training script, and a Streamlit app.

## Dataset

`email_spam_data.csv` — Contains 5,572 labeled messages (`ham` / `spam`). This is the classic SMS Spam Collection dataset (email spam classifiers are trained on this exact type of dataset).

## Project Structure

```text
email_spam_project/
├── email_spam_data.csv        # raw dataset (Label, Message)
├── email_spam_eda.py          # Step 1: EDA + Cleaning + Text Preprocessing + TF-IDF + Train-Test Split
├── train_model.py             # Step 2: Model training (Naive Bayes, Logistic Regression, XGBoost) + best model save
├── spam_prediction.py         # Step 3: Streamlit app (final deployment)
├── requirements.txt
└── README.md

```

## How to Run (in a conda environment)

```bash
# 1. Environment setup
conda create -n spam-predictor python=3.11 -y
conda activate spam-predictor
python -m pip install -r requirements.txt

# 2. Run EDA + preprocessing (this generates the TF-IDF vectorizer and train/test splits)
python -m email_spam_eda

# 3. Train the model (this saves the best model as spam_model.pkl)
python -m train_model

# 4. Run the Streamlit app
streamlit run spam_prediction.py

```

**Note:** The `email_spam_eda.py` file uses the path `/mnt/user-data/uploads/email_spam_data.csv` (just like in the original car price project). When running on your local system, replace this path with `email_spam_data.csv` (in the same folder).

## Pipeline Summary

1. **EDA**: Checked for class imbalance (~87% ham, ~13% spam), message length analysis, and top words per class.
2. **Cleaning**: Removed duplicates and empty messages.
3. **Text Preprocessing**: Lowercased text → removed URLs/emails/numbers/punctuation → removed stopwords → lemmatized.
4. **Feature Engineering**: TF-IDF vectorization (unigrams + bigrams, top 3,000 features).
5. **Modeling**: Compared Naive Bayes vs. Logistic Regression vs. XGBoost using the F1-score (rather than accuracy, due to the imbalanced dataset).
6. **Best Model**: Naive Bayes (F1 ~0.89, spam precision 100%) — auto-saved as `spam_model.pkl`.
7. **Deployment**: The Streamlit app accepts text input and delivers real-time spam/ham predictions along with a confidence score.

## Key Design Decisions (similar to the comments in the original project)

* Used a **stratified split** in `train_test_split` to preserve the spam/ham ratio across both training and testing sets.
* **TF-IDF was fitted only on the training data** and merely transformed on the test data to prevent data leakage (the same principle used with `StandardScaler` in the car price project).
* Made **F1-score** the primary metric for model selection.
