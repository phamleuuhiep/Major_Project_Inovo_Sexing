---
title: "In-Ovo Sexing Research - Planning 1"
date: 2025-09-25
tags: []
course: "Major Project"
---

# In-Ovo Sexing Research - Planning 1

## Task 1: Commercial Deployment of In-Ovo Sexing Technologies

**Assigned To:** Tam  
**Objective:** To identify and thoroughly research existing commercialized in-ovo sexing systems and comprehend their principles, performance, market impact, and the challenges they face in widespread deployment.

### Method

1. **Identify commercial systems and companies (16/9/2025 - 17/9/2025)**  
   - **Action:**  
     Understand the paper review (Section 3.3 *"Commercialization and Practical Applications"* and Table 7 *"Several in ovo sexing methods in marketing"*).  
     List all mentioned commercial products and the companies developing/marketing them.  
   - **Output:** A list of commercialized systems  

2. **Give details for each system's characteristics (17/9/2025 - 18/9/2025)**  
   - **Action:** For each identified system, extract the following crucial details:  
     - **Technology:** Core technology (e.g., VOCs + Terahertz, PCR, mass spectrometry, hyperspectral imaging, genome editing)  
     - **Incubation day:** Embryonic stage sexing is performed  
     - **Sample technique:** Sample extracted (allantoic fluid, embryonic cells) or non-invasive  
     - **Invasiveness:** Invasive or non-invasive; impact on commercialization  
     - **Metrics:** Accuracy / precision / recall  
     - **Throughput:** Eggs per hour (critical for hatcheries)  
     - **Marketing status:** Active / in development / under ethical review  
   - **Output:** Comparative table summarizing characteristics  

3. **Analyze Practical Applications and Challenges (18/9/2025 - 19/9/2025)**  
   - **Action:** Research practical implications:  
     - **Economic aspects:** Cost-efficiency, throughput (>20,000 eggs/hr), consumer/business impact  
     - **Ethical & welfare concerns:** Avoiding ~7B male chick culling, methods before day 7, genetic editing concerns  
     - **Technical limitations:** Infection risk, hatching impact, precision >98%, compatibility with hatchery trays  
     - **Facility integration:** Renovation, logistics, workforce requirements  
   - **Output:** Analysis of practicalities, benefits, and challenges with emphasis on key market requirements  

4. **Synthesize Findings (19/9/2025 - 20/9/2025)**  
   - **Action:** Consolidate insights highlighting:  
     - Leading commercial players  
     - Technologies used  
     - Deployment status  
     - Factors driving/hindering adoption  
   - **Output:** Research report on deployed in-ovo sexing products  

---

## Task 2: Biological and Genetic Foundations of Sex Differentiation

**Assigned To:** Hieu  
**Objective:** To understand the fundamental biological and genetic mechanisms that determine sex in chicken embryos, and explain how these intrinsic differences are utilized in in-ovo sexing technologies.

### Method

1. **Understand chicken sex chromosomes (16/9/2025 - 17/9/2025)**  
   - **Action:** Research avian ZZ/ZW system; males = ZZ, females = ZW.  
     Note ~2% DNA content difference between sexes.  
   - **Output:** Explanation of sex chromosome system  

2. **Explore early embryonic structures (17/9/2025 - 18/9/2025)**  
   - **Action:** Study blastoderm and imaging techniques (3D Micro-CT, MRI) for localization.  
   - **Output:** Description of blastoderm role and localization methods  

3. **Analyze hormonal differentiation & biomarkers (18/9/2025 - 19/9/2025)**  
   - **Action:** Study sex differentiation process, allantoic fluid (forms ~Day 5, max Day 12–13).  
     - **Key biomarker:** Estrone sulfate (measurable Day 9, >98% accuracy).  
     - Other hormones: Estrone (E1), Estradiol (E2), Estriol (E3), Testosterone (T), Androstenedione (A4), DHT.  
   - **Output:** Overview of hormones and biomarker dynamics  

4. **Investigate genetic marker-based techniques (19/9/2025 - 20/9/2025)**  
   - **Action:** Explain PCR targeting CHD genes on Z/W chromosomes; banding patterns.  
     Mention qRT-PCR, HRM improvements, flow cytometry for DNA content differences.  
   - **Output:** Explanation of PCR and flow cytometry methods  

5. **Summarize physiological changes timeline (20/9/2025)**  
   - **Action:** Chronological summary of markers:  
     - Day 2–3: Blastoderm growth, vascular/heartbeat  
     - Day 5–8: Gonad differentiation  
     - Day 9: Estrone sulfate in allantoic fluid  
     - Day 11–14: Feather color differences (breed-specific)  
     - Ethical note: Sexing before ~Day 7 (nervous system development)  
   - **Output:** Timeline summary  

---

## Task 3: Machine Learning and Deep Learning Approaches to Avian Sex Determination

**Assigned To:** Hiep  
**Objective:** To conduct an in-depth study of Machine Learning and Deep Learning models applied to in-ovo sexing classification, including preprocessing and performance analysis.

### Method

1. **Define AI’s role and data types (16/9/2025 - 17/9/2025)**  
   - **Action:** Establish binary classification problem, data acquisition, modeling.  
     - Spectral data (FT-IR, Raman, Hyperspectral, VIS/NIR)  
     - Morphological data (eggshell shape, blood vessel distribution)  
     - VOC profiles  
   - **Output:** Statement on AI role and data inputs  

2. **Detail preprocessing pipeline (17/9/2025 - 18/9/2025)**  
   - **Action:** Research smoothing methods:  
     - Moving Average, Savitzky-Golay (S-G), Multiple Scattering Correction (MSC), Differential Processing  
   - Feature extraction methods: PCA, ICA, SPA, CARS  
   - **Output:** Summary of smoothing & feature extraction techniques  

3. **Analyze ML models (18/9/2025 - 19/9/2025)**  
   - **Action:** Review ML methods & accuracies:  
     - SVM (63–80%), PLS-DA (up to 99%), ANN/MLP (~88% imaging, ~60% VOCs),  
     - Naïve Bayes / LR (~83%), LDA/KNN (~84%), RFC (VOCs),  
     - GA-BP / GA-ELM (~83–87%).  
   - **Output:** ML model descriptions, data types, performance  

4. **Investigate DL models (19/9/2025 - 20/9/2025)**  
   - **Action:** Review DL approaches:  
     - CNNs (VIS/NIR, ~94%)  
     - DBNs (blood vessel features, ~83%)  
     - Object Detection (Faster R-CNN, YOLO, SSD) for morphology tasks  
   - **Output:** DL model descriptions and benefits  

5. **Synthesize findings (20/9/2025)**  
   - **Action:** Compare ML vs DL strengths/weaknesses, discuss future research (XGBoost, CNNs, multiband databases).  
   - **Output:** Research report on AI models for sexing classification  

---
# In-Ovo Sexing Research - Planning 2 
##  Dataset Identification and Machine Learning Model Application

---

### Task 3.1: Available datasets 

The source materials detail a wide variety of data types derived from five major categories of in-ovo sexing techniques:  

- **Molecular-based**  
- **Spectral-based**  
- **Acoustic-based**  
- **Morphology-based**  
- **Volatile Organic Compound (VOC)-based** 

| Data Type Category                     | Specific Data/Features                                                                                                                                      | Source                                                                                                                             | Examples of Use/Collection                                                                      | Ref |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --- |
| **Morphology-Based Data (Image Data)** | Measurements of outer shape characteristics such as length, width, area, eccentricity, extent, and shape index. Used for determining sex before incubation. | Data collected from fertilized White Leghorn chicken eggs (n=143, 121 viable) using a low-cost smartphone-based imaging apparatus. | Also used for duck and Japanese quail eggs.                                                     |     |
| **Spectral/Imaging Data**              | **Blood Vessel Images (RGB):** Features from distribution of blood vessels (texture, geometry, HOG features).                                               | Images from Jingfen No. 6 breeding eggs (n=3024), days 3–5 incubation.                                                             | Suitable for white eggs due to light transmission.                                              |     |
|                                        | **Hyperspectral/VIS/NIR Data:** Near-UV (300–380 nm), visible (380–780 nm), near-IR (780–1500 nm). Features: hemoglobin pigment, feather color.             | Used for gender determination as early as day 7–8.                                                                                 | VIS-NIR spectroscopy: 99.52% accuracy on day 14 (Isa Brown eggs).                               |     |
|                                        | **FT-IR and Raman Spectroscopy:** Molecular fingerprint from pulp/blood cells.                                                                              | Used for sexing unincubated eggs via germinal disc.                                                                                | Raman discriminates sex using NIR fluorescence signals of blood flow in extraembryonic vessels. |     |
| **Molecular-Based Data**               | **Hormone Concentration:** Estrone sulfate in allantoic fluid.                                                                                              | Day 9 measurement.                                                                                                                 | Male eggs have lower estrone sulfate; >98% accuracy.                                            |     |
|                                        | **DNA/Genetic Markers:** Sex-specific genes (CHD). Chromosome content in nuclei.                                                                            | PCR-based methods.                                                                                                                 | Flow cytometry via DNA fluorescence.                                                            |     |
| **VOC Data**                           | Profiles of volatile compounds (nonanal, decanal, 6,10-dimethyl-5,9-undecadien-2-one).                                                                      | Collected during incubation.                                                                                                       | Prediction from Day 1 using SPME + GC-MS + E-nose.                                              |     |
| **Acoustic Data**                      | Heartbeat / heart rate signals.                                                                                                                             | After Day 9 incubation.                                                                                                            | Female embryos have 2–4 bpm higher heart rate than males after Day 17.                          |     |

## Task 3.2: Applying Machine Learning Models

### Focus Area 1: Morphology-Based Data (Low-Cost, Non-Invasive)

- **Wide Neural Network:**  
  - Accuracy: up to 88.9% (mean 81.5%).  
  - Captures non-linear relations; width & length most predictive features.  

- **Other ML Models:**  
  - Boosted Trees (83.8%), Cubic SVM (72.2%), Medium KNN (55.6%).  
  - Neural networks outperformed due to handling complex patterns.  

---

### Focus Area 2: Blood Vessel Image Data (Early Incubation)

- **Improved YOLOv7:**  
  - mAP 88.79% (90.32% female, 88.18% male).  
  - Enhancements: CBAM attention + BiFPN fusion → better accuracy & speed (23.9 ms/image).  

- **Other DL Models:**  
  - YOLOv5 (82.83%), Faster R-CNN (71.62%), SSD (56.30%).  
  - YOLOv7 significantly better.  

- **Fusion Models:**  
  - RF-DS (Random Forests + Dempster-Shafer) → 88.0% accuracy.  
  - Shows potential of multi-sensor fusion for higher accuracy.  

---

### Focus Area 3: General ML/DL Applications

- **ANN:** 82.86% accuracy (hyperspectral, Day 10, male classification).  
- **CNN:** VIS/NIR spectra + SPA algorithm → 93.83% accuracy (early duck embryos).  
- **Gaussian Naïve Bayes:** 82.88% accuracy (morphology features, quail eggs).  
- **LDA / Logistic Regression:** ~84% accuracy (light transmittance, Days 16–18).  
