# Membership-Inference Coding Task

## Overview

This project implements a membership inference attack on a pre-trained ResNet18 model. The goal is to classify whether a given sample is a member of the model's training dataset based on its features and outputs. This task involves feature extraction, model training, and prediction, followed by evaluation of the attack's performance using metrics like TPR@FPR=0.05 and AUC.

## Datasets
- **PUB Dataset**: Contains labeled samples (with `ids`, `imgs`, `labels`, and `membership` fields). Used for training and validating the membership inference model.
- **PRIV OUT Dataset**: Contains unlabeled samples (with `membership` field set to None). Predictions are made on this dataset to submit to the evaluation server.

## Workflow
The notebook follows these steps:

1. **Loading Datasets**:
   - PUB and PRIV OUT datasets are loaded, and their structure is examined.

2. **Model Preparation**:
   - A pre-trained ResNet18 model is loaded for feature extraction.

3. **Dataset Preprocessing**:
   - Custom datasets are created with PyTorch to handle data transformations and batching.

4. **Feature Extraction**:
   - Features are extracted from the PUB and PRIV OUT datasets using the ResNet18 model.

5. **Membership Inference Attack**:
   - A Random Forest Classifier is trained on the extracted features and corresponding membership labels from the PUB dataset.
   - Metrics such as TPR@FPR=0.05 and AUC are computed for the PUB dataset to validate the attack's performance.
   - Membership probabilities for the PRIV OUT dataset are predicted using the trained classifier.

6. **Submission**:
   - A CSV file containing `ids` and predicted membership probabilities (`scores`) for the PRIV OUT dataset is generated and submitted to the evaluation server.

7. **Request to the Server**:
	- The submission.csv is sent to the server.

	## Dependencies
The following libraries are required to run the notebook:
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Requests


## How to Run
1. Clone the repository and navigate to the project folder.
2. Make sure that the required dependencies are installed on your machine.
3. Open the Jupyter Notebook file (`membership_inference_attack.ipynb`).
4. Follow the code cells sequentially to:
   - Load the datasets.
   - Extract features and train the model.
   - Predict membership probabilities.
   - Generate and submit the results.

   ## File Structure
- `membership_inference_attack.ipynb`: Main Jupyter Notebook containing the code for the project.
- `pub.pt`: Training and validation dataset.
- `priv_out.pt`: Dataset for generating predictions.
- `submission.csv`: Generated CSV file for submission.
- `README.md`: Documentation for the project.


