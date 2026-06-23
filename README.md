# Ames House Price Prediction

Machine learning regression project for predicting residential property prices in Ames, Iowa.

This project was developed for Kaggle's **House Prices: Advanced Regression Techniques** competition. The goal is to predict the final sale price (`SalePrice`) of each house using a wide range of property characteristics, such as quality, size, location, age, garage information, and basement features.

## Project Goal

Build and compare regression models that predict a house's sale price as accurately as possible.

The main evaluation metric is:

- Root Mean Squared Error (RMSE)

A lower RMSE indicates more accurate predictions.

## Project Workflow

The project includes:

- Data exploration and descriptive analysis
- Missing-value analysis
- Visualization of numerical and categorical variables
- Correlation analysis with `SalePrice`
- Feature selection based on data insights
- Data preprocessing
- One-hot encoding for categorical features
- Standardization for numerical features
- Principal Component Analysis (PCA)
- Hyperparameter tuning
- Model comparison
- Kaggle submission generation

## Data Exploration

The notebook examines the distribution and predictive relevance of many housing features.

Important variables associated with higher sale prices included:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `GarageArea`
- `TotalBsmtSF`
- `1stFlrSF`
- `FullBath`
- `TotRmsAbvGrd`
- `YearBuilt`
- `YearRemodAdd`

Several features had substantial missingness or very limited predictive value. These were removed during preprocessing.

## Data Preprocessing

The preprocessing process included:

- Removing the identifier column
- Dropping features with a high percentage of missing values
- Removing rows with remaining missing values
- Removing low-information or noisy features
- Standardizing numerical variables
- One-hot encoding categorical variables

Examples of removed features include:

- `PoolQC`
- `MiscFeature`
- `Alley`
- `Fence`
- `MasVnrType`
- `FireplaceQu`
- `LotFrontage`
- `PoolArea`
- `ScreenPorch`
- `MiscVal`
- `MoSold`
- `YrSold`

## PCA Analysis

Principal Component Analysis was evaluated to reduce dimensionality.

- Original number of features: 72
- PCA components explaining 95% of the variance: 39

However, PCA reduced predictive performance for all evaluated models. Therefore, the final models used the cleaned feature set without PCA.

## Models Evaluated

The following regression models were trained and evaluated:

### Base Models

- Decision Tree Regressor
- K-Nearest Neighbors Regressor (KNN)
- Locally Weighted Linear Regression (LWLR)

### Ensemble Models

- Random Forest Regressor
- Bagging Regressor
- Gradient Boosting Regressor

## Hyperparameter Tuning

Hyperparameter tuning was performed using Grid Search and manual parameter search.

Examples of tuned parameters included:

- Decision Tree: maximum depth, minimum split size, and minimum leaf size
- KNN: number of neighbors, distance metric, and weighting strategy
- LWLR: bandwidth parameter `tau` and regularization parameter `lambda`

## Model Performance

| Model | Validation RMSE |
|---|---:|
| Decision Tree | 32,585.66 |
| KNN | 27,409.68 |
| Random Forest | 26,647.61 |
| Bagging Regressor | 25,807.08 |
| Gradient Boosting | 24,595.42 |
| **LWLR** | **22,998.03** |

## Final Model

The best-performing model was **Locally Weighted Linear Regression (LWLR)**.

Best parameters:

```python
tau = 2.0
reg_lambda = 0.01
