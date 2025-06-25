#  Loan Approval Prediction (Voting Classifier + MLflow)

This project predicts whether a loan should be approved or not using an ensemble VotingClassifier, with full experiment tracking via MLflow.
##  Project Structure
loan_approval_voting_classifier/
├── data/
├── notebooks/
├── scripts/
├── src/
├── images/
├── requirements.txt
├── .gitignore
└── README.md

##  How to Run
### 1. Start MLflow UI
```bash
start_mlflow_server.bat
run_training.bat
##  Features
- VotingClassifier (Decision Tree, Random Forest, Logistic Regression)
- CalibratedClassifierCV (Isotonic calibration)
- MLflow tracking:
  - Parameters
  - Metrics
  - Model artifacts
- Evaluation Metrics:
  - ROC Curve & AUC
  - Recall, Precision, F1 Score
## Requirements
bash
pip install -r requirements.txt
