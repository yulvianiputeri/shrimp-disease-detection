import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_preprocess_data
from model import bandingkan_model
from utils import extract_features_single

print("Starting model training process...")

X_train, X_test, y_train, y_test, scaler, pca = load_and_preprocess_data()
print(f"Data loaded and preprocessed. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

plt.figure(figsize=(10, 8))

X_pca_healthy = X_train[y_train == 0]
X_pca_diseased = X_train[y_train == 1]

plt.scatter(X_pca_healthy[:, 0], X_pca_healthy[:, 1], color='blue', label='Healthy', alpha=0.6)
plt.scatter(X_pca_diseased[:, 0], X_pca_diseased[:, 1], color='red', label='Diseased', alpha=0.6)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Feature Space of Shrimp Dataset (Training Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_clusters_augmented.png")
plt.close()
print("PCA scatter plot saved as pca_clusters_augmented.png.")


print("\nInitiating model training and comparison...")
hasil_model = bandingkan_model(X_train, y_train, X_test, y_test)

knn_model = hasil_model['KNN']['model']
svm_model = hasil_model['SVM']['model']

print("\nSaving trained models and preprocessing objects...")
# ============================================================
# VOTING ENSEMBLE
# ============================================================
print("\n" + "="*70)
print("🎯 TRAINING VOTING ENSEMBLE (KNN + SVM)")
print("="*70)

from model import train_voting_ensemble

acc_voting, report_voting, preds_voting, y_test_voting, voting_model = train_voting_ensemble(
    X_train, y_train, X_test, y_test, knn_model, svm_model
)

# Save voting model
print("\n💾 Saving voting_model.pkl...")
joblib.dump(voting_model, "voting_model.pkl")
print("✅ Voting model saved successfully!")

# ============================================================
# ACCURACY COMPARISON
# ============================================================
print("\n" + "="*70)
print("📊 ACCURACY COMPARISON")
print("="*70)
print(f"KNN Accuracy:              {hasil_model['KNN']['akurasi']:.4f} ({hasil_model['KNN']['akurasi']*100:.2f}%)")
print(f"SVM Accuracy:              {hasil_model['SVM']['akurasi']:.4f} ({hasil_model['SVM']['akurasi']*100:.2f}%)")
print(f"Voting Ensemble Accuracy:  {acc_voting:.4f} ({acc_voting*100:.2f}%)")
print("="*70)

# Check improvement
best_individual = max(hasil_model['KNN']['akurasi'], hasil_model['SVM']['akurasi'])
if acc_voting > best_individual:
    improvement = (acc_voting - best_individual) * 100
    print(f"\n✅ Voting Ensemble improved accuracy by {improvement:.2f}%")
elif acc_voting == best_individual:
    print("\n➡️  Voting Ensemble matched best individual model")
else:
    diff = (best_individual - acc_voting) * 100
    print(f"\n⚠️  Voting Ensemble is {diff:.2f}% lower than best individual model")

print("="*70 + "\n")