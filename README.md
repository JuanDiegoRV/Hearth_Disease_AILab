# Heart Disease Risk Prediction

This project implements logistic regression from first principles for heart-disease risk prediction. It covers exploratory analysis, a manual stratified split, normalization, binary cross-entropy optimization, decision boundaries, L2 regularization, and preparation for training and testing in Amazon SageMaker. The core model uses NumPy rather than scikit-learn.

## Dataset

The repository contains `heart.csv`, downloaded from the [John Smith Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). The assignment also references the [Neurocipher dataset page](https://www.kaggle.com/datasets/neurocipher/heartdisease). The downloaded file contains 1,025 rows, including 723 exact duplicates; the analysis removes those duplicates and documents the resulting 302 unique records.

## Requirements

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Run locally

1. Install the required libraries.
2. Keep `heart.csv` beside `heart_disease_lr_analysis.ipynb`.
3. Open the notebook and run all cells from top to bottom.

The notebook exports `sagemaker_train.csv` and `sagemaker_test.csv` using the same normalized split used for local evaluation.

## Main local result

The six-feature model reaches approximately 0.739 accuracy, 0.732 precision, 0.820 recall, and 0.774 F1 on the held-out test subset. Among the tested regularization strengths, `lambda = 1` reduces the weight norm and slightly lowers test binary cross-entropy without changing the threshold-based test metrics. These instructional results are not evidence of clinical validity.

## SageMaker training and testing

Upload the notebook, `heart.csv`, `sagemaker_train.csv`, and `sagemaker_test.csv` to AWS Academy SageMaker Studio. Run the notebook to train and test the model, record the available environment or instance configuration, and compare the resulting metrics with the local values above. Do not create or deploy an endpoint.

### Evidence collected

Current SageMaker Studio environment:

![SageMaker Studio environment](awsev1.jpeg)

Two additional screenshots are still required after cloud execution: one showing successful model training and another showing the held-out test metrics.
