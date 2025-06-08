# PCOS-Detection

This repository implements a supervised machine learning system for the automated classification of Polycystic Ovary Syndrome (PCOS) using structured clinical and biometric features. PCOS, a hormonal disorder characterized by ovarian dysfunction, metabolic anomalies, and hyperandrogenism, is typically diagnosed through combined biochemical, symptomatic, and ultrasound-based criteria. The system developed herein supports binary classification (PCOS vs non-PCOS) using feature-driven modeling.

Data Ingestion and Feature Engineering:

* Input dataset: Derived from Kaggle’s publicly available PCOS dataset (or equivalent), comprising structured patient-level features, including:

  * Demographic attributes (age, weight, height, BMI)
  * Physiological and endocrinal metrics (blood pressure, follicle count, hormone levels)
  * Secondary features (menstrual irregularity, hair growth score, skin darkening, etc.)
* Null handling: Imputation via mean/median strategy (continuous) and mode encoding (categorical)
* Categorical encoding: Binary and ordinal attributes mapped using label encoding
* Feature scaling: Standardization (z-score normalization) or MinMax scaling depending on model compatibility

Model Architecture:

* Input vector: X ∈ ℝⁿ, where n = number of preprocessed features
* Feed-forward neural network with:

  * Dense(input\_dim, 128) → ReLU
  * Dropout(p=0.3)
  * Dense(128, 64) → ReLU
  * Dropout(p=0.3)
  * Dense(64, 32) → ReLU
  * Dense(32, 1) → Sigmoid

Training Configuration:

* Loss function: Binary cross-entropy
* Optimizer: Adam (learning rate = 1e-3)
* Evaluation metrics: Binary accuracy, AUC-ROC, F1-score, precision, recall
* Batch size: 32
* Epochs: 100 with early stopping (monitor: val\_loss, patience = 10)
* Cross-validation: Stratified k-fold (k = 5) for robustness against class imbalance

Performance Metrics:

* Test accuracy: >90% (depending on feature subset and model variant)
* AUC-ROC: Typically >0.92 in cross-validated results
* Precision and recall evaluated to monitor false negative rate (critical in clinical screening)

Baseline Comparisons:

* Benchmark models trained for comparison:

  * Logistic Regression (L2 regularization)
  * Random Forest (n=100, max\_depth=10)
  * SVM with RBF kernel
* Deep learning model consistently outperforms traditional ML in F1-score and generalization on unseen data

Deployment Pipeline:

* Trained model exported in HDF5 (`model.h5`) format
* Inference API compatible with Flask/FastAPI for web-based deployment
* Feature normalization must match training-time scaler (stored using joblib or pickle)

Dependencies:

* Python 3.8+
* TensorFlow / Keras
* scikit-learn
* pandas, numpy, seaborn, matplotlib (for EDA and visualization)

Use Cases:

* Clinical decision support systems for gynecologists and endocrinologists
* Integration into telemedicine diagnostic platforms
* Educational tool for medical ML modeling
