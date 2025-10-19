Simple Disease Predictor (Machine Learning ) 🩺🧪

This is a single-file, self-contained Python script that demonstrates a full machine learning classification pipeline using scikit-learn.

The project is designed to simulate a basic health diagnostic system, classifying a patient's condition into one of three simulated diseases based on 5 numerical symptom scores.

Project Features ✨

Synthetic Data Generation: Uses sklearn.datasets.make_classification to create 1,000 synthetic patient records.

5 Features (Symptoms): The model is trained on five features: Fever_Severity, Cough_Frequency, Fatigue_Level, Headache_Intensity, and Body_Ache.

Random Forest Classifier: A robust ensemble model is trained for classification.

Model Evaluation: Outputs the classification accuracy and a detailed classification_report.

Sample Prediction: Demonstrates how to use the trained model to predict the disease for a new, unseen patient sample.

**Prerequisites 🛠️
**
To run this script, you must have Python installed, along with the following libraries:

pip install scikit-learn pandas numpy


How to Run ▶️

Save the code as simple_disease_predictor.py.

Execute the script from your terminal:

python simple_disease_predictor.py


The output will display the data setup details, model performance metrics, and the prediction result for the sample patient data.
