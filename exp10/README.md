# Experiment 10: Single-Layer Artificial Neural Network (ANN)

## Overview

This experiment implements a single-layer Artificial Neural Network for
binary classification using Python and NumPy. It uses one sigmoid output
neuron, binary cross-entropy loss, and gradient descent to learn weights
and bias. The notebook uses the Breast Cancer Wisconsin dataset and
selects **mean radius** and **mean texture** for a compact two-feature
demonstration. This is an educational experiment, not a medical
diagnostic tool.

## Aim

To build and implement a single-layer Artificial Neural Network using
Python to solve a simple classification problem.

## Objectives

-   Understand the structure and working of a single-layer ANN.
-   Implement a neural-network neuron using Python and NumPy.
-   Train the model on labeled data.
-   Use sigmoid activation to calculate probabilities.
-   Update weights and bias using gradient descent.
-   Predict test and new-sample class labels.
-   Evaluate performance using classification metrics and
    visualizations.

## Relevance and Applications

A single-layer ANN is a foundation for understanding neurons, weights,
bias, activation functions, loss, and learning. Related
binary-classification applications include spam detection, pass/fail
prediction, basic document categorization, and quality-control
classification. Real deployments require suitable validation and
domain-specific safeguards.

## Theory

### Artificial Neural Network

An ANN is a machine-learning model inspired by interconnected processing
units. Each artificial neuron combines inputs with trainable weights and
a bias, then applies an activation function. Training changes parameters
to reduce prediction error.

### Single-layer ANN

This notebook uses input features connected directly to one output
neuron, with **no hidden layer**. For two features: \[
z=w_1x_1+w_2x_2+b=`\mathbf{x}`{=tex}\^{T}`\mathbf{w}`{=tex}+b \] where x
is the input vector, w is the weight vector, and b is the bias.

### Sigmoid activation

\[ `\sigma`{=tex}(z)=`\frac{1}{1+e^{-z}}`{=tex} \] The output is between
0 and 1 and is interpreted as the estimated probability of class 1. With
a threshold of 0.5, probabilities at least 0.5 predict class 1; lower
probabilities predict class 0. In this dataset, class 0 is malignant and
class 1 is benign.

### Binary cross-entropy loss

\[
L=-`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}`\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]`{=tex}\]
Here, N is the number of observations, y is the true binary label, and p
is the predicted probability. Lower loss indicates better fit to the
training labels, but does not alone guarantee good test performance.

### Gradient descent

For sigmoid output with binary cross-entropy: \[
`\frac{\partial L}{\partial \mathbf{w}}`{=tex}=`\frac{1}{N}`{=tex}`\mathbf{X}`{=tex}\^{T}(`\mathbf{p}`{=tex}-`\mathbf{y}`{=tex})
\] \[
`\frac{\partial L}{\partial b}`{=tex}=`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}(p_i-y_i)
\] Updates are: \[
`\mathbf{w}`{=tex}`\leftarrow`{=tex}`\mathbf{w}`{=tex}-`\eta`{=tex}`\frac{\partial L}{\partial\mathbf{w}}`{=tex},`\quad`{=tex}
b`\leftarrow `{=tex}b-`\eta`{=tex}`\frac{\partial L}{\partial b}`{=tex}
\] where η is the learning rate. Training repeats over multiple epochs.

### Feature scaling

Standardization transforms a feature using the training-set mean and
standard deviation: \[
x\_{`\mathrm{scaled}`{=tex}}=`\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}`{=tex}
\] The scaler is fitted only on the training set and then applied to the
test set to avoid data leakage and help optimization.

### Dataset

The Breast Cancer Wisconsin dataset is loaded using
`sklearn.datasets.load_breast_cancer`. The notebook selects
`mean radius` and `mean texture` as input features. The target is
binary: malignant or benign. Using two features makes the decision
boundary easier to visualize but may limit predictive performance
compared with a well-validated model using more information.

## Algorithm

1.  Import NumPy, Pandas, Matplotlib, and scikit-learn utilities.
2.  Load and inspect the dataset.
3.  Check missing values, duplicates, and class distribution.
4.  Select two input features and the target.
5.  Visualize the selected features.
6.  Split data into training and testing sets with stratification.
7.  Standardize using statistics learned from the training data.
8.  Initialize weights and bias.
9.  Calculate the weighted sum and sigmoid probabilities.
10. Calculate binary cross-entropy.
11. Calculate gradients and update weights and bias.
12. Repeat training over multiple epochs and plot loss.
13. Predict test probabilities and convert them to labels using
    threshold 0.5.
14. Evaluate using accuracy, precision, recall, F1-score, and confusion
    matrix.
15. Plot decision regions and predict a new illustrative sample.

## Implementation Details

The network is implemented explicitly using NumPy: - `weights`:
trainable feature weights. - `bias`: trainable bias term. - `sigmoid()`:
converts weighted sums into probabilities. - `binary_cross_entropy()`:
calculates average loss. - The training loop performs forward
propagation, gradient calculation, and parameter updates.

Scikit-learn provides the dataset, train-test split, standardization,
and evaluation metrics. The notebook uses a fixed random seed for a
reproducible split. The learning rate and epoch count are
hyperparameters and may affect the result.

## Evaluation Metrics

Class 1 (benign) is treated as the positive class: - **Accuracy:**
fraction of all test observations predicted correctly. - **Precision:**
among observations predicted benign, fraction actually benign. -
**Recall:** among actual benign observations, fraction correctly
predicted benign. - **F1-score:** harmonic mean of precision and
recall. - **Confusion matrix:** summarizes correct and incorrect
predictions for both classes. - **Training loss curve:** shows binary
cross-entropy over epochs.

Consider these metrics together because different classification errors
can have different consequences.

## Decision Boundary

A single sigmoid neuron without a hidden layer has a linear threshold
decision boundary in the selected feature space. The notebook plots the
predicted regions using a grid of standardized feature values. A linear
boundary cannot represent every possible pattern.

## Expected Outputs

-   Dataset preview, shape, and feature information
-   Data-quality checks and class distribution plot
-   Feature scatter plot
-   Training/test sizes and standardized values
-   Learned weights and bias
-   Training-loss values and plot
-   Predicted probabilities and labels
-   Accuracy, precision, recall, F1-score, and classification report
-   Confusion matrix and decision-region plot
-   Demonstration prediction for a new sample

Use actual values from your executed notebook when reporting results.

## Advantages

-   Relatively easy to understand and implement.
-   Demonstrates the core learning components of an ANN.
-   NumPy makes the mathematics visible.
-   Useful for learning gradient descent and binary cross-entropy.
-   Fast on small datasets.

## Limitations

-   No hidden layer; the model learns only a linear decision boundary.
-   Performance depends on feature selection and training settings.
-   Two features may not contain enough information for the best
    classification.
-   One train-test split gives only one estimate of generalization.
-   Probabilities are not automatically well calibrated.
-   This educational example must not be used for medical decisions.

## Conclusion

This experiment implements a single-layer ANN for binary classification
using NumPy. The sigmoid function produces probabilities, and gradient
descent learns weights and bias by minimizing binary cross-entropy.
Classification metrics, a confusion matrix, training-loss curve, and
decision-boundary plot help evaluate and interpret the model.

## How to Run

1.  Open the notebook in Jupyter Notebook or JupyterLab.

2.  Install dependencies if required:

    ``` bash
    pip install numpy pandas matplotlib scikit-learn jupyter
    ```

3.  Select the environment containing the required packages.

4.  Run all cells from top to bottom.

## References

-   [Breast Cancer Wisconsin
    dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
-   [StandardScaler
    API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
-   [Scikit-learn model
    evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
-   [Reference notebook: Single-layer neural
    networks](https://github.com/rasbt/pattern_classification/blob/master/machine_learning/singlelayer_neural_networks/singlelayer_neural_networks.ipynb)
