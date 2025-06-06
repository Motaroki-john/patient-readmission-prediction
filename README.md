# Patient Readmission Risk Prediction

This project predicts hospital readmission risk for diabetic patients using a logistic regression model. It includes data preprocessing, exploratory data analysis (EDA), model training, and visualization of results.

## Files
- `diabetic_data.csv`: Dataset used for training.
- `hospital_readmission_prediction.py`: Main script to preprocess data, train the model, and generate visualizations.
- `readmission_distribution.png`, `readmission_by_age.png`, `feature_importance.png`, `confusion_matrix.png`: Visualizations generated by the script.

## How to Run
1. Clone the repository.
2. Create a Conda environment: `conda create -n readmission_env python=3.9`
3. Install dependencies: `conda install pandas scikit-learn matplotlib seaborn; pip install joblib`
4. Run the script: `python hospital_readmission_prediction.py`