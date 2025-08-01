# Lung Cancer Prediction using Machine Learning

This project was developed as part of the M.Sc. course “Advanced Storage, Processing, and Retrieval of Big Data” at Western Michigan University. The aim is to build a predictive machine learning model that can identify the risk of lung cancer using health-related and lifestyle survey data.

## 🧠 Problem Statement
Lung cancer is one of the most fatal diseases worldwide. Early detection significantly increases the chance of treatment success. This project applies supervised machine learning models to predict lung cancer risk based on 15 binary health indicators and one continuous variable (age).

## 📊 Dataset
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- Size: 309 entries, 16 features
- Features include: Smoking, Yellow Fingers, Chronic Disease, Wheezing, Chest Pain, etc.
- Target: Lung Cancer (Yes/No)

## 🛠️ Technologies & Libraries
- Python (NumPy, Pandas, Matplotlib, Seaborn)
- Scikit-learn (KNN, SVM, preprocessing)
- LightGBM
- Imbalanced-learn
- Microsoft Azure (Cloud deployment - optional)

## 🔍 Models Evaluated
| Model                  | Precision | Sensitivity | F1-Score |
|-----------------------|-----------|-------------|----------|
| K-Nearest Neighbors   | 1.00      | 0.86        | 0.93     |
| Support Vector Machine| 1.00      | 0.98        | 0.99     |
| LightGBM Classifier   | 1.00      | 0.90        | 0.95     |

**Support Vector Machine** (SVM) was selected for deployment due to its highest sensitivity.

## 🧪 How to Run
1. Open the `Lung_cancer_detection.ipynb` notebook in **Google Colab**.
2. Upload the dataset `survey_lung_cancer.csv` under the `data/` folder.
3. Follow the code blocks sequentially to:
   - Clean and preprocess data
   - Train ML models
   - Evaluate model performance
   - Input new data and get predictions
4. Optional: Deploy the SVM model in cloud via Microsoft Azure Automation.

## 📁 Folder Structure
lung-cancer-prediction-ml/
├── data/
├── notebooks/
├── reports/
├── app/
├── README.md
├── requirements.txt


## 📈 Future Improvements
- Add more training data
- Experiment with deep learning methods
- Build an interactive web app (e.g., Streamlit)
- Improve generalizability with cross-validation and better imbalance handling

## 👨‍💻 Contributors
- Ifrat Zaman
- Gnana Deepak Madduri
- Shubham Pawar

## 📄 License
This project is for academic and learning purposes only.
