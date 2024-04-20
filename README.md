# breast-cancer-detection

This project uses supervised and unsupervised learning to train a model. Which can be used to predict and classify future values to see if patients have breast cancer. This repository contains Jupyter notebooks for evaluating machine learning models using various techniques. The notebooks include implementations for k-nearest neighbors (KNN), random forest, and long short-term memory (LSTM) models.

## Files:

1. k-nearest-neighbor.ipynb
   - Evaluates a KNN model using the k-nearest neighbors algorithm.
   - Implements model evaluation using k-fold cross-validation and computes performance metrics such as accuracy, precision, recall, F1 score, and confusion matrix.
2. random-forest.ipynb
   - Implements a Random Forest classifier and evaluates its performance.
   - Computes accuracy, precision, recall, F1 score, and confusion matrix for model evaluation.
3. lstm.ipynb
   - Implements a Long Short-Term Memory (LSTM) model for sequence prediction.
   - Evaluates the LSTM model's performance using accuracy, precision, recall, F1 score, and confusion matrix.

## Dataset:

All notebooks use the same dataset provided in the file data.csv. This file contains data from the Wisconson Breast Cancer research.
Which can be found at [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download)

## Instructions:

1. Ensure you have the necessary dependencies installed, including pandas, scikit-learn, keras, and numpy.
2. Place the data.csv file in the same directory as the Jupyter notebooks.
3. Execute the code cells in each notebook sequentially to train and evaluate the respective machine learning models.
4. Review the output metrics and analysis provided in each notebook to assess the performance of the models.

## Results:

The notebooks provide detailed insights into the performance of each machine learning model, including metrics such as accuracy, precision, recall, F1 score, and confusion matrix.

For further details, refer to the individual notebooks.

If you have any questions or suggestions, feel free to reach out. Happy modeling!
