# Iris Dataset Analysis and KNN Classification

This project involves the analysis of the Iris dataset and the implementation of a K-Nearest Neighbors (KNN) classifier to predict the species of iris flowers based on their features.

## Project Structure

The project is organized into several steps, each represented by a cell in the Jupyter Notebook:

1. **Import Libraries**: Import necessary libraries such as `numpy`, `pandas`, `seaborn`, `matplotlib`, and `sklearn`.

2. **Load and Prepare Data**: Load the Iris dataset and prepare it for analysis by creating a DataFrame and mapping species labels.

3. **Data Exploration**: Display the first few rows of the dataset and create various plots to visualize the data:
    - Boxplots for petal length and petal width by species.
    - Scatter plots for sepal length vs. sepal width and petal length vs. petal width, colored by species.
    - Pairplot to visualize relationships between all features.
    - Correlation heatmap to show the correlation between features.

4. **Train-Test Split**: Split the data into training and testing sets.

5. **KNN Classifier**: Implement and train a KNN classifier:
    - Train the model with a specified number of neighbors.
    - Predict the labels for the test set.
    - Calculate and print the accuracy of the model.

6. **Hyperparameter Tuning**: Use GridSearchCV to find the best hyperparameters for the KNN classifier:
    - Define a parameter grid.
    - Perform grid search with cross-validation.
    - Print the best parameters and the best score.
    - Evaluate the best model on the test set.

## Dependencies

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

## Usage

To run the project, open the Jupyter Notebook and execute the cells in order. The notebook will guide you through the steps of loading the data, visualizing it, training the KNN classifier, and tuning its hyperparameters.

## Results

The project demonstrates the use of KNN for classification and the impact of hyperparameter tuning on model performance. The best model achieved an accuracy of 100% on the test set.

## License

This project is licensed under the MIT License.# iris