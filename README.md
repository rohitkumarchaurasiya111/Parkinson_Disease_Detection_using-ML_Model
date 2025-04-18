# Parkinson's Disease Detection using Machine Learning

This project leverages a Machine Learning model to detect Parkinson’s Disease based on biomedical voice measurements. The primary objective is to classify whether a person is affected by Parkinson's Disease using various features extracted from voice recordings.

## 🧠 Overview

Parkinson’s Disease is a neurodegenerative disorder that affects movement. Early diagnosis is crucial for effective treatment and management. This project uses a dataset of voice measurements and applies machine learning algorithms to accurately predict the presence of the disease.

## 📁 Project Structure

- `Parkinsons_Disease_Prediction.ipynb` – Jupyter Notebook containing all the code for data loading, preprocessing, model training, evaluation, and prediction.
- `parkinsons.data` – Dataset used for training and testing the model.
- `README.md` – Documentation for the project.

## 📊 Dataset

The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons). It contains 195 voice recordings from 31 people, 23 of whom have Parkinson's Disease.

**Features include:**
- MDVP:Fo(Hz) – Average vocal fundamental frequency
- MDVP:Jitter(%), MDVP:RAP, etc. – Several measures of variation in frequency
- MDVP:Shimmer, APQ3, etc. – Measures of amplitude variation
- NHR, HNR – Measures of noise
- RPDE, DFA – Nonlinear dynamical complexity measures
- spread1, spread2, D2 – Nonlinear measures of fundamental frequency variation
- `status` – 1 indicates the presence of Parkinson's, 0 indicates a healthy individual

## ⚙️ Model Building Process

1. **Data Loading**: Loaded using pandas.
2. **Exploratory Data Analysis (EDA)**: Checked for missing values, feature distributions, and correlations.
3. **Feature Scaling**: Used StandardScaler to normalize data.
4. **Model Selection**: Trained a Support Vector Machine (SVM) classifier.
5. **Model Evaluation**: Evaluated the model using accuracy score, confusion matrix, and classification report.
6. **Prediction**: Built a simple interface to test predictions on new data.

## ✅ Results

- **Accuracy**: Achieved high accuracy on the test set.
- **Precision/Recall/F1-score**: The model demonstrated strong performance in classifying both Parkinson's and non-Parkinson's cases.

## 🛠️ Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/rohitkumarchaurasiya111/Parkinson_Disease_Detection_using-ML_Model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Parkinson_Disease_Detection_using-ML_Model
   jupyter notebook Parkinsons_Disease_Prediction.ipynb
   ```
3. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

## 🚀 Future Work

- 🧪 **Model Optimization**: Apply hyperparameter tuning (GridSearchCV or RandomizedSearchCV) to further improve accuracy and performance.
- 🧠 **Try Other Models**: Explore and compare performance using different algorithms such as Random Forest, Gradient Boosting, XGBoost, and Neural Networks.
- 🌐 **Web App Deployment**: Build a user-friendly interface using Streamlit or Flask to allow users to upload data and get predictions in real-time.
- 📲 **Mobile Integration**: Convert the model into an API and integrate it into a mobile application for easier accessibility.
- 📉 **Feature Engineering**: Investigate new features or apply dimensionality reduction techniques like PCA to improve model generalization.
- 🗃️ **Larger Dataset**: Train the model on a larger and more diverse dataset to improve robustness and reduce overfitting.
- 📊 **Visualization Dashboard**: Create an interactive dashboard for visualizing predictions, metrics, and patient history.

## 👩‍💻 Author

**Rohit Kumar Chaurasiya**  
💼 Aspiring Data Scientist & Machine Learning Enthusiast  
📧 Email: rohitkumarchaurasiya111@gmail.com
🔗 GitHub: [rohitkumarchaurasiya111](https://github.com/rohitkumarchaurasiya111)  
📍 Nepal  


## 🤝 Contributing

Contributions are welcome! If you have suggestions, improvements, or want to build on this project, feel free to fork the repo and create a pull request.
