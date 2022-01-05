# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a random forest classifier from the sklearn library. It was trained using default parameters (hyperparameter tuning considered outside of the scope of the project given we focus on the MLOps aspect of the project)

## Intended Use

The intended use of the model is to classify whether the salary of a given individual is above or below $50000

## Training Data

The training data is the census bureau data. More information on the dataset can be found here: https://archive.ics.uci.edu/ml/datasets/census+income. 80% of the data was used for training while the remaining 20% were used for evaluation

## Evaluation Data

The evaluation data was based on 20% of the census bureau data mentioned above.

## Metrics
The following metrics were used with their respective score:
- Precision: 0.72
- Recall: 0.62
- Fbeta: 0.67


## Ethical Considerations
The main ethical considerations were to evaluate our model on different groups found in the data. Sex and race being two important factors, we see that there is no drastic difference in performance between each groups (see slice_output.txt). However, there were some large discrepancy when it comes to working class for example, something we should maybe explore in more details in the future.

## Caveats and Recommendations
There was little focus on the modelling side of things given we have focused on the MLOps part of the project.
In order to improve the model performance we could perform feature engineering and feature selection on the dataset, try different models and select best one, and perform hyperparameter tuning if necessary.
