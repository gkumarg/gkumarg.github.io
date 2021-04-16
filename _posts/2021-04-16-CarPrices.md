# Car Price Prediction
End to End deployment of ML model to predict car prices for a dataset from Kaggle.

### Steps in Jupyter notebook:
1. Downloaded kaggle dataset.
2. Conducted exploratory data analysis on the dataset.
3. Looked for duplicates, missing data, outliers etc.
4. Created a new feature for number of years based on year of the car using current year as the reference.
5. Used one-hot encoding for categorical features like fuel type and automatic/manual gear etc.
6. Removed extra variables that are not needed like year of the car since we have a new feature for it.
7. Split the dataset into train and test.
8. Used different hyperparameters for cross-validation using RandomForestRegressor model to find the best parameters.
9. Created a final model.
10. Predictions were tested using the final model.
11. Exported the model as a pickle file for deployment.

You can access the app that I created from here:
[Gopakumar's Car Price Prediction App](https://indian-car-price-prediction.herokuapp.com/)

Also checkout the full github link for the [files](https://github.com/gkumarg/carpricepredictions).
