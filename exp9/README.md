# Experiment 09: K-Means Clustering

## Overview

This experiment implements K-Means, an unsupervised machine-learning
algorithm that groups observations according to feature similarity. The
notebook uses the Iris dataset and selects petal length and petal width.
Species labels are excluded from training and may be used only for
post-clustering evaluation.

## Aim

To implement K-Means clustering in Python, partition data into groups
based on feature similarity, and analyze the clusters using suitable
evaluation methods.

## Objectives

-   Understand the K-Means algorithm.
-   Load and inspect a dataset.
-   Select and standardize numerical features.
-   Examine candidate cluster counts using the Elbow Method.
-   Train K-Means and assign cluster labels.
-   Visualize clusters and centroids.
-   Evaluate clusters using WCSS, silhouette score, and optional
    external comparison with known labels.

## Relevance and Applications

Clustering is useful when data has no labels or when discovering natural
groupings. Applications include customer segmentation, document
grouping, image segmentation, market analysis, product grouping, and
exploratory data analysis.

## Theory

### What is clustering?

Clustering is an unsupervised learning task that groups similar
observations. Similarity depends on the selected features and distance
measure.

### K-Means

K-Means partitions observations into K clusters. Each cluster is
represented by its centroid, the mean of the observations assigned to
that cluster. It alternates between: 1. **Assignment:** assign each
observation to its nearest centroid. 2. **Update:** recalculate each
centroid as the mean of its assigned observations.

The process repeats until assignments or centroids stabilize, or an
iteration limit is reached.

### Euclidean distance

For observation x and centroid μ: \[
d(x,`\mu`{=tex})=`\sqrt{\sum_{j=1}^{m}(x_j-\mu_j)^2}`{=tex} \] where m
is the number of features.

### WCSS / inertia

K-Means minimizes the within-cluster sum of squares: \[
WCSS=`\sum`{=tex}*{k=1}\^{K}`\sum`{=tex}*{x_i`\in `{=tex}C_k}\|x_i-`\mu`{=tex}\_k\|\^2
\] In scikit-learn, `KMeans.inertia_` reports this sum of squared
distances. WCSS usually decreases as K increases, so it should not be
used alone to select an arbitrarily large K.

### Elbow Method

Fit models for several K values and plot K against WCSS. A bend in the
curve may indicate a useful trade-off between compact clusters and model
complexity. The elbow is a heuristic, not a guaranteed unique answer.

### Silhouette score

The silhouette score measures how well an observation fits its own
cluster compared with other clusters. It ranges from -1 to 1: values
near 1 suggest well-separated clusters, values near 0 suggest a
boundary, and negative values can indicate a potentially unsuitable
assignment. Interpret it alongside plots and knowledge of the data.

### Feature scaling

Distance-based algorithms can be dominated by features with larger
numerical ranges. Standardization transforms a feature using: \[
z=`\frac{x-\mu}{\sigma}`{=tex} \] The notebook uses `StandardScaler`.
Centroids can be converted back to original units using
`inverse_transform`.

### Dataset

The Iris dataset from `sklearn.datasets.load_iris` contains 150
observations, four numerical measurements, and three species. This
experiment uses only **petal length** and **petal width**. Species
labels must not be included in the K-Means input; if compared with
clusters later, that is external evaluation only.

## Algorithm

1.  Import libraries and load Iris.
2.  Inspect records, dimensions, statistics, missing values, and
    duplicates.
3.  Select petal length and petal width.
4.  Visualize the selected features.
5.  Standardize the features.
6.  Fit K-Means for candidate K values and record WCSS.
7.  Plot the Elbow Method.
8.  Calculate silhouette scores for candidate values.
9.  Fit the final model (K=3 in the notebook).
10. Assign cluster labels and count observations per cluster.
11. Visualize observations and centroids.
12. Convert centroids back to original feature units.
13. Evaluate the result using silhouette score and, where included, ARI
    and a cluster/species cross-tabulation.

## Implementation Details

Libraries: NumPy, Pandas, Matplotlib, and scikit-learn. The notebook
uses `init="k-means++"`, `n_init=10`, and `random_state=42`. K-means++
helps initialize centroids, multiple initializations reduce dependence
on one random start, and the fixed random state supports
reproducibility.

## Evaluation

-   **WCSS/inertia:** within-cluster compactness.
-   **Elbow plot:** compares the reduction in WCSS as K changes.
-   **Silhouette score:** measures cohesion and separation.
-   **Adjusted Rand Index (ARI):** when known labels are available,
    compares cluster assignments with reference labels while adjusting
    for chance.
-   **Cross-tabulation:** shows how reference species are distributed
    among clusters.

Cluster IDs such as 0, 1, and 2 are arbitrary; a cluster number does not
inherently represent a particular species.

## Expected Outputs

-   Dataset preview, structure, statistics, and data-quality checks
-   Scatter plot of selected features
-   WCSS table and Elbow plot
-   Silhouette scores and plot
-   Cluster labels and cluster counts
-   Cluster plot with centroids
-   Centroids in original feature units
-   Evaluation scores and optional cluster/species comparison

Report actual metric values from the executed notebook rather than
assuming values in advance.

## Advantages

-   Simple to understand and implement.
-   Efficient for many numerical datasets.
-   Centroids provide compact cluster summaries.
-   Useful as a baseline clustering method.

## Limitations

-   K must be selected in advance.
-   Results can depend on initialization.
-   Sensitive to scaling and outliers.
-   Works best for compact, roughly spherical clusters.
-   May struggle with overlapping or irregularly shaped groups.
-   Cluster IDs have no inherent semantic meaning.

## Conclusion

This experiment demonstrates how K-Means groups numerical observations
without using class labels during training. Feature standardization,
WCSS analysis, the Elbow Method, silhouette scoring, centroid
visualization, and optional external evaluation illustrate the main
clustering workflow.

## How to Run

1.  Open the notebook in Jupyter Notebook or JupyterLab.

2.  Install dependencies if required:

    ``` bash
    pip install numpy pandas matplotlib scikit-learn jupyter
    ```

3.  Select the environment containing these packages.

4.  Run all notebook cells from top to bottom.

## References

-   [Scikit-learn clustering
    guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)
-   [KMeans
    API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
-   [Iris
    dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
-   [Silhouette
    Coefficient](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
