"""
Email Spam Prediction - Model Training & Evaluation
======================================================
email_spam_eda.py chalane ke baad ye script use hoga.
Teen models compare karenge: Multinomial Naive Bayes, Logistic Regression, XGBoost
Best model (F1-score ke basis par) ko save karenge, kyunki dataset imbalanced hai
(sirf accuracy dekhna misleading hoga).
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# =========================================================
# STEP 1: PROCESSED DATA LOAD KARO
# =========================================================
print("STEP 1: Loading processed TF-IDF data")
X_train = sp.load_npz('/home/claude/X_train_tfidf.npz')
X_test = sp.load_npz('/home/claude/X_test_tfidf.npz')
y_train = pd.read_csv('/home/claude/y_train.csv').squeeze()
y_test = pd.read_csv('/home/claude/y_test.csv').squeeze()

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Spam ratio in train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

# =========================================================
# STEP 2: MULTIPLE MODELS TRAIN + COMPARE KARO
# =========================================================
print("\n" + "="*60)
print("STEP 2: TRAINING & COMPARING MODELS")
print("="*60)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric='logloss', random_state=42,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # imbalance handle karo
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1_Score": f1})
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("F1_Score", ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON (sorted by F1-Score)")
print("="*60)
print(results_df.to_string(index=False))
# Insight: Spam class chhota hai (~13%), isliye F1-score sabse fair metric hai
# accuracy akela misleading ho sakta hai (sab ko 'ham' predict karke bhi ~87% accuracy mil jayegi)

# =========================================================
# STEP 3: BEST MODEL SELECT KARO
# =========================================================
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\nBest Model (highest F1): {best_model_name}")

# =========================================================
# STEP 4: CONFUSION MATRIX (best model)
# =========================================================
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('/home/claude/confusion_matrix.png', dpi=100)
plt.close()

# Model comparison bar chart
plt.figure(figsize=(9, 5))
results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].plot(kind='bar')
plt.title('Model Comparison')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('/home/claude/model_comparison.png', dpi=100)
plt.close()

# =========================================================
# STEP 5: BEST MODEL SAVE KARO (deployment ke liye)
# =========================================================
print("\n" + "="*60)
print("STEP 5: SAVING BEST MODEL")
print("="*60)

if best_model_name == "XGBoost":
    best_model.save_model('/home/claude/spam_model.json')
    print("XGBoost model saved as spam_model.json")
else:
    joblib.dump(best_model, '/home/claude/spam_model.pkl')
    print(f"{best_model_name} model saved as spam_model.pkl")

# Model type ko bhi save karo taaki streamlit app ko pata rahe kaunsa format load karna hai
with open('/home/claude/model_info.txt', 'w') as f:
    f.write(best_model_name)

results_df.to_csv('/home/claude/model_comparison_results.csv', index=False)

print("\n" + "="*60)
print("ALL STEPS COMPLETE! Best model saved.")
print("="*60)
