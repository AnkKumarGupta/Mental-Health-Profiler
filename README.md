# Mental Disorder Detection using Cognitive Distortions

### An Explainable AI Framework for Depression Risk Assessment


## Problem Statement
Standard AI models for mental health often act as "Black Boxes," flagging risk without explaining *why*. This project builds a **"White Box"** system that mimics clinical therapy (CBT) by:
1.  Detecting specific **Cognitive Distortions** (thinking errors like *Overgeneralization*).
2.  Using these patterns to predict **Depression Severity** (Minimum, Mild, Moderate, Severe).

---

## Datasets

This project utilizes two distinct datasets to train the specialist models for each stage.

### **Dataset 1: Cognitive Distortion Detection Detection (Stage 1)**
* **Source:** [Kaggle -  Annotated_data](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset) (Shreevastava & Foltz, 2021)
* **Description:** A csv file consisting four columns *Patient Questions*, *Distorted Part*, *Primary Distortion*, *Secondary Distortion*.
* **Size:** ~1,597 samples.
* **Labels:** 10 specific Cognitive Distortion types (e.g., *Catastrophizing*, *Labeling*, *Mental Filtering*) plus "No Distortion".

### **Dataset 2: Depression Severity (Stage 2)**
* **Source:** [GitHub - DepSeverity](https://github.com/usmaann/Depression_Severity_Dataset) (Naseem et al., 2022)
* **Description:** A csv file consisting of Social media posts (Reddit) curated to analyze depression levels.
* **Size:** *3,553* samples.
* **Labels:** 4-level severity scale: *Minimum (2,587), Mild (290), Moderate (394), Severe (282)*.

## Architecture
This system uses a **Two-Stage Hybrid Pipeline**:

### **Stage 1: The Specialist (MentalBERT)**
* **Model:** `mental-bert-base-uncased` fine-tuned on the Cognitive Distoriton Detection dataset.
* **Task:** Multi-label classification of 10 Cognitive Distortion types.
* **Technique:** Weighted Full Fine-Tuning (to handle class imbalance).
* **Output** Whether the input text is normal or have cognitive distortion along with which type of cognitive distortion.

### **Stage 2: The Profiler (XGBoost)**
* **Input:** A 16-dimensional Hybrid Vector combining:
    * 11 Distortion Probabilities (from Stage 1).
    * 5 Sentiment/Metadata features (VADER Sentiment + Word Count).
* **Technique:** Gradient Boosting with **SMOTE** (Synthetic Minority Over-sampling) to balance the severe risk class.
* **Output:** Severity Risk Level + Clinical Rationale.

---

## Results & Benchmarks

Our Hybrid approach outperforms established academic baselines.

| Task | Our Model | 
| :--- | :--- | 
| **Distortion Type Detection** | **0.38** (Weighted F1) | 
| **Severity Estimation** | **0.62** (Weighted F1) | 

---

## Demo

The final part of the code uses **ipywidgets** to create a user friendly interface that takes input and produces the output in a very simple and intuitive way. It can be used after executing the file *Training_Pipeline_with_Demo.ipynb*

## Reference

1. Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2022). "MentalBERT: Publicly Available Pre-trained Language Models for Mental Healthcare." Proceedings of LREC 2022, 7184–7190.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 16, 321–357.
3. Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International AAAI Conference on Weblogs and Social Media.
4. https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset?resource=download  (**Dataset for Stage 1 - Annotated_data.csv**)
5. Naseem et al. "Early Identification of Depression Severity Levels on Reddit Using Ordinal Classification" Proceedings of the ACM Web Conference 2022, 2563--2572 (https://github.com/usmaann/Depression_Severity_Dataset)  (**Dataset for Stage 2 - Reddit_depression_dataset.csv**)


## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/AnkKumarGupta/Mental-Health-Profiler.git
cd Mental-Health-Profiler

```

### 2. Run the code
It will have a code file named as *Training_Pipeline_with_Demo.ipynb*, just run it and enjoy the results.

