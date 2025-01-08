---
title: Fraud Detection ML App
emoji: 🏆
colorFrom: green
colorTo: purple
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---


# Model description
This is a Gaussian Naive Bayes model trained on a synthetic dataset, containining a large variety of transaction types representing normal activities as well as 
abnormal/fraudulent activities generated by J.P. Morgan AI Research. The model predicts whether a transaction is normal or fraudulent.

## Intended uses & limitations
For educational purposes

## Training Procedure
The data preprocessing steps applied include the following:
- Dropping high cardinality features. This includes Transaction ID, Sender ID, Sender Account, Beneficiary ID, Beneficiary Account, Sender Sector 
- Dropping no variance features. This includes Sender LOB
- Dropping Time and date feature since the model is not time-series based
- Transforming and Encoding categorical features namely: Sender Country, Beneficiary Country, Transaction Type, and the target variable, Label
- Applying feature scaling on all features
- Splitting the dataset into training/test set using 85/15 split ratio
- Handling imbalanced dataset using imblearn framework and applying RandomUnderSampler method to eliminate noise which led to a 2.5% improvement in accuracy


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6662300a0ad8c45a1ce59190/BEi0CfOfJ2ytxD5VoN4IM.png)

## Model Plot
![image](https://github.com/saifhmb/Fraud-Detection-ML-App/assets/111028776/f9c30bf5-3036-4397-a0e7-693205b39154)


## Evaluation Results

| Metric   |    Value |
|----------|----------|
| accuracy | 0.794582 |

### Model Explainability
SHAP was used to determine the important features that helps the model make decisions
![image](https://github.com/user-attachments/assets/824937c9-9290-40c2-8aba-ac908f5f1028)



### Confusion Matrix
![image](https://github.com/saifhmb/Fraud-Detection-ML-App/assets/111028776/e03a40b9-0196-4df4-a251-fb1c701c1a56)




