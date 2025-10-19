# -*- coding: utf-8 -*-
"""
 Python Machine Learning Project: Simple Disease Predictor

This script uses the scikit-learn library to create a classification model.
It generates a synthetic dataset resembling a health diagnosis problem (based on features),
trains a Random Forest classifier, evaluates its performance, and makes a sample prediction.

To run this script, you must have the following libraries installed:
pip install scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification # Used for simple data generation

# --- 1. DATA GENERATION AND PREPARATION ---

# Define the features (symptoms) and target classes (diseases)
# --- UPDATED: Added a 5th feature (Body_Ache) ---
FEATURE_NAMES = ['Fever_Severity', 'Cough_Frequency', 'Fatigue_Level', 'Headache_Intensity', 'Body_Ache']
CLASS_LABELS = ['No Disease', 'Flu', 'Common Cold']

print("--- Data Generation and Setup ---")

# Generate a synthetic dataset for a 3-class classification problem (3 diseases)
# X: Features (symptoms)
# y: Target (disease class indices 0, 1, 2)
X, y = make_classification(
    n_samples=1000,          # Total number of patient records
    n_features=5,            # --- UPDATED: 5 Number of features (symptoms) ---
    n_informative=5,         # All features are informative
    n_redundant=0,           # No redundant features
    n_classes=3,             # Number of unique target classes (diseases)
    n_clusters_per_class=1,  # Data clusters per class
    random_state=42          # Seed for reproducibility
)

# Convert the NumPy arrays to a Pandas DataFrame for easier viewing/handling
df = pd.DataFrame(X, columns=FEATURE_NAMES)
df['Disease_Index'] = y

print(f"Generated a dataset with {len(df)} samples and {len(FEATURE_NAMES)} features.")

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples\n")


# --- 2. MODEL TRAINING ---

print("--- Model Training (Random Forest) ---")

# Initialize the Random Forest Classifier
# Random Forest is a robust ensemble method suitable for simple classification tasks.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

print("Model training complete.\n")


# --- 3. MODEL EVALUATION ---

print("--- Model Evaluation ---")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy:.4f}\n")

# Display a detailed classification report
print("Classification Report:")
# Target names are mapped from the index (0, 1, 2) to the actual disease names
print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))


# --- 4. MAKING A NEW PREDICTION ---

print("--- Sample Prediction ---")

# Example patient data (symptom scores are standardized/normalized for the model)
# The array must now contain 5 values corresponding to the 5 FEATURE_NAMES
# Sample 1: High Fever, Medium Cough, Low Fatigue, High Headache, High Body Ache
new_patient_data = np.array([[2.0, 0.5, -1.0, 1.5, 1.8]]) # Added 1.8 for Body_Ache

# Prediction
predicted_index = model.predict(new_patient_data)[0]
predicted_disease = CLASS_LABELS[predicted_index]

# Prediction probability (confidence)
probabilities = model.predict_proba(new_patient_data)[0]
confidence = probabilities[predicted_index] * 100

# Displaying the result
print(f"\nNew Patient Symptoms (Normalized): {new_patient_data[0]}")
print(f"The model predicts the patient has: '{predicted_disease}'")
print(f"Confidence: {confidence:.2f}%")

# Another sample: Low severity symptoms
# Sample 2: Low Fever, Low Cough, Low Fatigue, Low Headache, Low Body Ache (Likely 'No Disease')
new_patient_data_low = np.array([[-1.5, -0.5, -2.0, -0.8, -1.2]]) # Added -1.2 for Body_Ache
predicted_index_low = model.predict(new_patient_data_low)[0]
predicted_disease_low = CLASS_LABELS[predicted_index_low]
print(f"\nAnother sample prediction (Low Symptoms): '{predicted_disease_low}'")

# --- 5. FEATURE IMPORTANCE (Bonus) ---

print("\n--- Feature Importance ---")
# Determine which features were most influential in the model's decision-making
feature_importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

print("Features ranked by importance in the model:")
print(feature_importances)

# 
