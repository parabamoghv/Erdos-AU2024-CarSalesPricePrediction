# Erdos-AU2024-CarSalesPricePrediction
Erdos Institute Data Science project on car sales price prediction


Repository structure: 

Erdos-AU2024-CarSalesPricePrediction/  
│  
├── data/                   # Data folder  
│   ├── raw/                # Raw data from Kaggle  
│   └── processed/          # Cleaned and processed data for analysis  
│  
├── notebooks/              # Jupyter notebooks for analysis and experimentation  
│   ├── 01_data_cleaning.ipynb    # Data cleaning notebook   
│   ├── 02_data_exploration.ipynb    # Data exploration notebook  
│   ├── 03_feature_engineering.ipynb      # Feature selection/engineering  
│   ├── 04_model_MLR.ipynb      # MLR model  
│   └── #add more models  
│    
├── src/                    # Source code for the project. This is the final product.  
│   ├── __init__.py         # Makes src a package  
│   ├── main.py             # Main script for running the project. Run this to predict prices using our model.  
│     
│  
│  
│  
├── report/                # Reports and generated outputs  
│   ├── figures/            # Images or plots generated during analysis  
│   └── final_report.pdf     # Final report or summary of findings  
│  
├── README.md               # Project overview and instructions  
├── .gitignore              # Files and directories to be ignored by Git. At a later point ignore data.  
└── LICENSE                 # License GNU  





# Modeling the relationship between car sales price and different car features

Buying and selling cars is a common experience, especially among people living in rural areas with little or no transportation. It is thus interesting to study what factors influence car sales prices significantly and what can be improved to have better sales and fair car sales prices. Using different Machine Learning Regression methods, we will develop models to predict car sales prices using the [CarDekho Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho/data?select=Car+details+v3.csv).

## Authors

- [Mariana Khachatryan](https://github.com/mariana-khach)
- [Adreja Mondol](https://github.com/adrejamondol)
- [Amogh Parab](https://github.com/parabamoghv)
- [Nasim Dehghan](https://github.com/nasimdeh)


## KPI

#### Technical performance Key Performance Indicators (KPIs)

1.	Mean Absolute Error (MAE)
2.	Mean Squared Error (MSE)
3.	Root Mean Squared Error (RMSE)
4.	R-squared- proportion of variance in target variable that is predictable from input features

#### Business impact KPIs

These KPIs measure how effectively the model adds value to the business, such as improving pricing strategies or increasing customer satisfaction.
1.	Revenue Increase: Accurate price predictions could help optimize the selling price of cars, leading to better margins.
2.	Cost Reduction: If the model is used to automate pricing, it can reduce the need for manual evaluation and pricing, saving labor costs.
3.	Inventory Turnover Rate: How quickly cars are being sold after their prices are set by the model.

## Key Stakeholders

1.	Car Dealerships need price prediction model to set competitive and accurate prices for cars. Dealerships want to maximize profit while ensuring quick car sales. Accurate price prediction results in competitive pricing and profitability.
2.	Customers can use the model to estimate whether the set price is fare.



## Exploratory Data Analysis and Feature Engineering

In this project we use data from CarDekho. Founded in 2008, it is India's leading online marketplace that helps individuals and dealers buy, sell, and manage their cars.
The data contains 8128 entries and 12 features corresponding to car name, year bought, selling price, kilometers driven, fuel type, seller type, transmission type, owner type, mileage, engine in the units of cubic capacity (CC), max_power in the units of brake horsepower (bhp), torque and number of seats. We didn't use torque feature since the units used for different entries correspond to different physical quantities and some of them have fixed values while others show range of values. We also dropped car name feature, since it had too many unique categories.
- Data cleaning involved:
  - removal of duplicated rows, which dropped entries to 6926 (removed 14.79% of data)
  - removal of 209 rows with missing values for multiple columns, which reduced entries to 6717 (3.02% less statistics)
  - removal of 1.28% of remaining data corresponding to "LPG" and "CNG" underrepresented gas fuel types which left 6631 entries
  - removal of 0.08% data corresponding to "Test Drive Car" of owner category which left 6626 entries
  - removal of outliers (5.42% of remaining data) 
- Final data set had 6533 data points with 9 features 
- Feature engineering
  - we modified number of seats feature to have two bins for number of seats >5 and seats $\leq 5$ 
    ![Nseats](https://github.com/user-attachments/assets/cdad7d03-64a4-4c7b-9972-2c52f4d147f7)
  - we combined "Trustmark Dealer" with "Dealer" categories in seller type feature, to decrease imbalance between different categories
    ![dealer](https://github.com/user-attachments/assets/5c4fc41a-5d9e-4938-99bf-83f3be569d2a)
    
We then used one hot encoding for categorical features (fuel type, seller type, transmission type, owner type, number of seats bin category).
- Exploratory data analysis:
  - we have looked at the correlation of sales price with different feature
    ![salesprice_corr](https://github.com/user-attachments/assets/0a7ef427-fb65-4007-8880-aa1d838f95f8)
  - sales price has highest correlation with max power
    ![salesprice_maxpower](https://github.com/user-attachments/assets/56d2330a-7fff-4608-9568-8eba2f533b17)
  - Before model training, correlated features were removed, i.e. only one feature among features with correlation $\geq 8$ was kept.
    ![2D_corr](https://github.com/user-attachments/assets/708235f7-1d9a-4bb5-bf5b-d44f5c26655a)

We can see the distributions of sales price for differnt categorical features in the violin plots below. the distributions look different for different categories of given categorical feature, which suggests that they should all be included in training of model.
![price_fuel](https://github.com/user-attachments/assets/fa84906e-8be6-43e2-8f2e-0108747e7b26)
![price_sellertype](https://github.com/user-attachments/assets/5daee49e-c9b9-4240-bb7b-5302d84650f5)
![price_transmission](https://github.com/user-attachments/assets/122ebd1a-0053-402b-89ff-6fc17216dd83)
![price_Nseats](https://github.com/user-attachments/assets/14c60686-74af-4156-872e-69eb2db58c56)
![price_owner](https://github.com/user-attachments/assets/94adbec0-f007-41a0-91f8-ebd918d129c2)


## Approach

We compare predictions of different Machine Learning models:
- Linear Regression
- Tree Methods
- Support Vector Machines
We start with linear regression model as a baseline model. We then compare results from different regression models. We use grid search to find optimal model parameters for different regression methods. 
We want to learn:
- Which features have the most influence on sales price prediction?
- Which regression model will provide the best performance?
- What is the prediction error?
The model framework is displayed in the following diagram:

![Modeling_framework](https://github.com/user-attachments/assets/4c78a63e-e484-48b6-94c2-d4fa86745b79)

As we are trying different regression models including one's that use distance metric, we use one-hot encoding for categorical features and scale the data for features using Standard scaling.
We also divide label values by 10^5 to have it in the units of 100000 Indian Rupees.


## Linear Regression (Base model)

As a baseline model we us simple linear regression (LinearRegression model from python sklearn library). We get Root Mean Squared Error (RMSE) of $1.35$ and $R^2=0.72$.
We the calculate residuals (difference between predicted label values and true values) and plot it vs true values to check Homoscedasticity of data. We can see in the picture below that the assumption of homoscedasticity is violated.

![Homoscedasticity](https://github.com/user-attachments/assets/9b7a8521-33b3-46bf-8537-5f4f0d43df24)

We also check normality by plotting residuals vs theoretical quartiles of normal distribution. 

![Normality](https://github.com/user-attachments/assets/9a793392-75b3-4671-b57b-e1ad20eb7708)


We can see that normality is violated for lower for residual values below -3.
These results indicate that we should consider other non-linear models.
We also did more advance analysis using Principal Component Analysis to reduce number of features. Using Grid Search Cross-Validation to tune model parameters,
we found that using PCA doesn't improve the performance of Linear Regresssion.

## Extreme Gradient Boosting (XGBoost)

Overall best model performance was obtained wit XGBoost with RMSE of $0.866$ and $R^2=0.88$. 
The performance of different models is summarized in the following table.

![image](https://github.com/user-attachments/assets/767913ad-0ab1-4d4a-981e-1d2f9345051a)



XGBoost outperforms SVMs and kNN because it is inherently nonlinear and robust to scale and is less sensitive to hyperparameter tuning.  
XGBoost improves performance by combining multiple trees, which enhances it's ability to model complex patterns. It also reduces overfitting by combining multiple trees and employing shrinkage/regularization.

## SHapley Additive exPlanations (SHAP values) for describing feature importances

SHAP values is a method based on cooperative game theory (https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137). SHAP shows the contribution or the importance of each feature on the prediction of the model.
The SHAP values for one single observation are shown in the following plot:

![Waterfall_onerow](https://github.com/user-attachments/assets/49765a99-3903-4c4b-b159-9a675c48cba7)

x-axis has the values of car sales price. x is the chosen observation, f(x) is the predicted value of the model, given input x and E[f(x)] is the mean of all predictions.
The SHAP value for each feature in this observation is given by the length of the bar. The sum of all SHAP values will be equal to E[f(x)] — f(x).
For analysis of the global effect of the features we can look at he following plots.

![Feature_shapvalues](https://github.com/user-attachments/assets/65965062-18f5-4f46-8c2d-c0f83d58739d)

Here the features are ordered from the highest to the lowest effect on the prediction. It takes in account the absolute SHAP value, so it does not matter if the feature affects the prediction in a positive or negative way.
We can see that the three features that have the most effect on the model prediction are year, max_power and engine. This is consistent with our results from EDA. 

Following violin plot also shows the global effect of the features on model prediction. Here we can also see how higher and lower values of the feature will affect the result.

![Feature_shapvalues_violin](https://github.com/user-attachments/assets/919c84af-e08a-4873-8583-e31acca550f0)


