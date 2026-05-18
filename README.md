# Employee Salary Prediction with KNN

This project is a Jupyter Notebook implementation of a KNN classification model that predicts employee salary categories using the `adult.csv` dataset.

## Files

- `Employee Salary Prediction.ipynb` - main notebook with data loading, preprocessing, model training, and evaluation
- `adult.csv` - dataset used by the notebook
- `requirements.txt` - Python package dependencies

## Setup

1. Create and activate a Python virtual environment

   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook in Jupyter or VS Code and run the cells.

## Notes

- The notebook handles missing values, label encodes categorical features, scales numerical features, and trains a `KNeighborsClassifier`.
- The model is evaluated using accuracy, confusion matrix, and classification report.
