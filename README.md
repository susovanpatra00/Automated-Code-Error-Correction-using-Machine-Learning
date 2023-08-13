# Automated Code Error Correction using Machine Learning

This project focuses on automatically correcting errors in programming code submissions using machine learning techniques. The goal is to predict the most likely error types in erroneous C code snippets.

## Summary

- Built a multiclass classification model using Multinomial Logistic Regression to predict 50 different error types in C code snippets
- Extracted bag-of-words features from raw code tokens and handled class imbalance in the training data
- Developed a Python program to load sparse matrices, train the Logistic Regression model, and make predictions on new unseen code snippets
- Tuned hyperparameters like max_iter and solver to optimize model performance
- Achieved 87.4 % accuracy in predicting error types on test data

## Implementation 

The main Python scripts are:

- `data.py`: Functions to load the sparse matrix data
- `model.py`: Logistic Regression model training and prediction
- `predict.py`: Driver script to load data, train model, and run predictions

The trained Logistic Regression model is serialized in `model.pkl`.

The main libraries used are NumPy, SciPy, and scikit-learn.

## Usage

To train the model:

```
python predict.py --train
``` 

To run predictions on new data: 

```
python predict.py --predict NEW_DATA
```

## References
- Relevant research papers and resources on automated program repair and error correction.
- Purushottam Kar (IITK)
