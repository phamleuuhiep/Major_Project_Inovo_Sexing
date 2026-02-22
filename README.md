🐣 Inovo Sexing Project – Pretest Notebook
1. Overview

This repository contains the pretest and feature engineering notebook for the Inovo Sexing project.

The purpose of this phase is to:

Validate dataset integrity

Perform preprocessing and reconstruction (if necessary)

Extract core shape-based morphological features

Conduct sanity checks and outlier filtering

Prepare a structured dataset for baseline modeling

This notebook represents the data validation and feature extraction stage before model development.

2. Scope of This Notebook
✔ Shape Feature Finalization

The following geometric descriptors are extracted:

Area

Major Axis Length

Minor Axis Length

Aspect Ratio

Eccentricity

Shape Index (SI)

These features serve as interpretable morphological predictors.

✔ Data Quality & Validation

The notebook includes:

Dataset length verification

Label merging validation

Feature completeness checks

Outlier detection

Reconstruction rate analysis

Sanity checks on segmentation output

❌ Not Included in This Phase

Spectral features

Deep learning models

Advanced augmentation

Hyperparameter optimization

This stage focuses purely on clean feature engineering and structural reliability.

3. Pipeline Structure
Raw Data
   ↓
Preprocessing
   ↓
Mask / Shape Extraction
   ↓
Feature Engineering
   ↓
Sanity Check & Cleaning
   ↓
Final Structured Dataset
4. Feature Description
Feature	Description
Area	Pixel area of segmented region
Major Axis Length	Longest axis of fitted ellipse
Minor Axis Length	Shortest axis of fitted ellipse
Aspect Ratio	Major Axis / Minor Axis
Eccentricity	Ellipse eccentricity (0–1)
Shape Index (SI)	Curvature-based shape descriptor
5. Validation Strategy

The notebook ensures:

Correct sample count (e.g., 257 samples)

No mismatch between labels and features

No empty feature vectors

Logical value ranges (e.g., aspect ratio ≥ 1)

Removal of extreme or corrupted samples

Statistical summaries are generated before exporting the cleaned dataset.

6. Output Artifacts

The notebook produces:

Cleaned feature dataframe

Summary statistics

Reconstruction statistics

Diagnostic logs

Optional exported CSV for modeling

7. Environment Requirements
Python Version

Python 3.9+

Core Libraries
numpy
pandas
matplotlib
scikit-learn
opencv-python
scikit-image

Install dependencies:

pip install -r requirements.txt
8. Next Phase Roadmap

After completing this notebook:

Exploratory Data Analysis (EDA)

Feature normalization / scaling

Train–test split

Baseline models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

Cross-validation

Feature importance analysis

Performance comparison

9. Design Philosophy

This pretest stage prioritizes:

Interpretability

Reproducibility

Explicit validation

Clean and explainable feature pipeline

The goal is to ensure the dataset is structurally sound and statistically reliable before moving to modeling and optimization.

10. Status

✔ Feature extraction complete
✔ Sanity checks implemented
🔜 Ready for EDA and baseline modeling
