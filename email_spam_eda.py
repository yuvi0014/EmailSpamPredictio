"""
Email Spam Prediction - EDA, Data Cleaning, Text Preprocessing, Train-Test Split
===================================================================================
Target Variable: Label (spam / ham)
Message (raw email/SMS text) ko clean karke TF-IDF features banayenge.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# =========================================================
# STEP 1: DATA LOAD KARO
# =========================================================
df = pd.read_csv('/mnt/user-data/uploads/email_spam_data.csv')
print("STEP 1: Data Loaded")
print(f"Shape: {df.shape}")
print(df.head())
print()

# =========================================================
# STEP 2: EDA (Exploratory Data Analysis)
# =========================================================
print("="*60)
print("STEP 2: EDA")
print("="*60)

# 2.1 Basic info - data types, nulls
print("\n--- Data Types & Nulls ---")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
# Insight: Dataset mein koi missing value nahi hai

# 2.2 Duplicate rows check karo
print("\n--- Duplicates ---")
print(f"Duplicate rows: {df.duplicated().sum()}")
# Insight: Kaafi duplicate messages mile (forwarded spam texts) - cleaning ke time hatayenge

# 2.3 Class distribution (imbalance check)
print("\n--- Label Distribution ---")
print(df['Label'].value_counts())
print(df['Label'].value_counts(normalize=True) * 100)
# Insight: Dataset imbalanced hai (~87% ham, ~13% spam)
# -> Model evaluation mein sirf accuracy pe bharosa nahi karenge, precision/recall/F1 dekhenge

plt.figure(figsize=(6, 5))
sns.countplot(x='Label', data=df, palette=['steelblue', 'salmon'])
plt.title('Ham vs Spam Distribution (Imbalanced)')
plt.tight_layout()
plt.savefig('/home/claude/label_distribution.png', dpi=100)
plt.close()

# 2.4 Message length feature - EDA insight ke liye
df['Message_Length'] = df['Message'].apply(len)
df['Word_Count'] = df['Message'].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x='Message_Length', hue='Label', kde=True, ax=axes[0], bins=50)
axes[0].set_title('Message Length by Label')
sns.boxplot(x='Label', y='Word_Count', data=df, ax=axes[1])
axes[1].set_title('Word Count by Label')
plt.tight_layout()
plt.savefig('/home/claude/length_analysis.png', dpi=100)
plt.close()
# Insight: Spam messages generally lambe hote hain ham messages ki tulna mein
# (spam mein promotional text, links, offers zyada words consume karte hain)

# 2.5 Most common words in spam vs ham (quick raw look, before cleaning)
def get_top_words(text_series, n=15):
    words = ' '.join(text_series).lower().split()
    words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation)]
    return pd.Series(words).value_counts().head(n)

print("\n--- Top raw words in SPAM messages ---")
print(get_top_words(df[df['Label'] == 'spam']['Message']))
print("\n--- Top raw words in HAM messages ---")
print(get_top_words(df[df['Label'] == 'ham']['Message']))
# Insight: Spam mein "free", "call", "text", "won", "claim" jaise words zyada frequent hain

# =========================================================
# STEP 3: DATA CLEANING
# =========================================================
print("\n" + "="*60)
print("STEP 3: DATA CLEANING")
print("="*60)

df_clean = df.copy()

# 3.1 Duplicate rows hatao
before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates(subset=['Message'])
print(f"Duplicates removed: {before - df_clean.shape[0]} rows "
      f"({before} -> {df_clean.shape[0]})")

# 3.2 Empty / very short messages check
df_clean = df_clean[df_clean['Message'].str.strip().str.len() > 0]
print(f"Shape after removing empty messages: {df_clean.shape}")

# =========================================================
# STEP 4: TEXT PREPROCESSING (NLP CLEANING)
# =========================================================
print("\n" + "="*60)
print("STEP 4: TEXT PREPROCESSING")
print("="*60)

def clean_text(text):
    text = text.lower()                                   # lowercase
    text = re.sub(r'http\S+|www\.\S+', ' ', text)          # URLs hatao
    text = re.sub(r'\S+@\S+', ' ', text)                   # email addresses hatao
    text = re.sub(r'\d+', ' ', text)                       # numbers hatao (phone/amount)
    text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation hatao
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

print("Cleaning text: lowercase -> remove URLs/emails/numbers/punctuation -> "
      "remove stopwords -> lemmatize")
df_clean['Clean_Message'] = df_clean['Message'].apply(clean_text)
print(df_clean[['Message', 'Clean_Message']].head())

# Encode target: ham=0, spam=1
df_clean['Label_Encoded'] = df_clean['Label'].map({'ham': 0, 'spam': 1})

print(f"\nFinal cleaned shape: {df_clean.shape}")

# =========================================================
# STEP 5: TRAIN-TEST SPLIT
# =========================================================
print("\n" + "="*60)
print("STEP 5: TRAIN-TEST SPLIT")
print("="*60)

X = df_clean['Clean_Message']
y = df_clean['Label_Encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("Stratified split use kiya taaki spam/ham ratio train aur test dono mein same rahe "
      "(class imbalance ki wajah se)")

# =========================================================
# STEP 6: TF-IDF VECTORIZATION
# =========================================================
print("\n" + "="*60)
print("STEP 6: TF-IDF VECTORIZATION")
print("="*60)

# NOTE: fit sirf training data par, transform dono par -> taaki data leakage na ho
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF feature matrix (train): {X_train_tfidf.shape}")
print(f"TF-IDF feature matrix (test): {X_test_tfidf.shape}")

# Save cleaned + processed datasets
df_clean.to_csv('/home/claude/email_spam_data_cleaned.csv', index=False)

import scipy.sparse as sp
sp.save_npz('/home/claude/X_train_tfidf.npz', X_train_tfidf)
sp.save_npz('/home/claude/X_test_tfidf.npz', X_test_tfidf)
y_train.to_csv('/home/claude/y_train.csv', index=False)
y_test.to_csv('/home/claude/y_test.csv', index=False)

import joblib
joblib.dump(tfidf, '/home/claude/tfidf_vectorizer.pkl')
print("\nTF-IDF vectorizer saved (needed later for prediction on new messages)")

print("\n" + "="*60)
print("ALL STEPS COMPLETE! Files saved.")
print("="*60)
