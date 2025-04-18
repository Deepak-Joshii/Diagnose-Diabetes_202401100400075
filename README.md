🧠 Diabetes Diagnosis using Machine Learning

This project aims to predict whether a person is diabetic based on medical diagnostic measurements using supervised machine learning algorithms. It utilizes the `2. Diagnose Diabetes.csv` dataset and demonstrates a full ML pipeline — from preprocessing and EDA to model training and evaluation.

---

## 📂 Dataset Overview

The dataset contains **768 patient records** with **8 input features** and **1 output label**:

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes probability based on family history |
| `Age` | Age (in years) |
| `Outcome` | Target variable (0 = Non-diabetic, 1 = Diabetic) |

---

## ⚙️ Methodology

1. **Data Cleaning**  
   - Replaced zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with **median** values to handle missing or implausible data.

2. **Feature Scaling**  
   - Applied **StandardScaler** to normalize all features for improved model performance.

3. **Exploratory Data Analysis (EDA)**  
   - Visualized feature distributions and outliers using histograms and boxplots  
   - Analyzed correlations using heatmaps  
   - Compared feature distributions across diabetic and non-diabetic classes

4. **Model Training**  
   - Split the dataset into **80% training** and **20% testing**  
   - Trained three classification models:  
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)  
     - Random Forest

5. **Model Evaluation**  
   - Evaluated performance using:  
     - Accuracy, Precision, Recall, F1-Score  
     - Confusion Matrix  
     - ROC-AUC Curve  

---

## 📈 Results Snapshot

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression |   --     |    --     |   --   |    --    |   --    |
| K-Nearest Neighbors |   --     |    --     |   --   |    --    |   --    |
| Random Forest       |   --     |    --     |   --   |    --    |   --    |

> Replace `--` with actual metrics after running the code.

---

## 💻 Installation & Requirements

### 📦 Required Libraries

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
