# 🏠 Boston Housing Price Prediction

A beginner-friendly Machine Learning project based on the **Boston Housing Dataset**.

In this project, I explored residential housing data from Boston, Massachusetts, and learned how different property and area characteristics can be used to estimate house prices.

The main purpose of this project was to **learn and practice the complete basic Machine Learning workflow** — from understanding and cleaning the data to building and evaluating a Linear Regression model.

---

## 📌 About the Project

Imagine working for a real estate development company in Boston.

Before starting a residential project, the company wants to estimate the possible value of a property based on characteristics such as:

* Number of rooms
* Distance from employment centres
* Accessibility to highways
* Pollution level
* Pupil-teacher ratio
* Percentage of lower-income population
* Whether the property is next to the Charles River
* Other characteristics of the area

I used these features to build a **Multivariable Linear Regression model** for estimating residential property prices.

> **Note:** This is a learning project based on historical data. The dataset represents Boston-area housing data from the 1970s and should not be treated as a model for current Boston property prices.

---

## 🎯 What I Learned

Through this project, I practiced:

* Loading a CSV dataset using Pandas
* Understanding rows, columns and features
* Checking for missing values
* Checking for duplicate records
* Understanding descriptive statistics
* Data exploration and manipulation
* Data visualization
* Understanding relationships between variables
* Splitting data into training and testing sets
* Building a Multivariable Linear Regression model
* Understanding regression coefficients
* Understanding predictions and residuals
* Checking residual distributions
* Understanding skewness
* Applying Log Transformation
* Comparing the original and transformed models
* Evaluating model performance using R²
* Using a trained model to estimate a property's value

---

## 🛠️ Technologies & Libraries

* **Python**
* **Pandas** – data loading, cleaning and manipulation
* **NumPy** – numerical calculations and transformations
* **Matplotlib** – data visualization
* **Seaborn** – statistical visualizations
* **Plotly** – interactive visualization
* **Scikit-learn** – Machine Learning and Linear Regression
* **Jupyter Notebook**

---

## 📊 Dataset

The dataset contains **506 observations** and **13 predictive features**, with `PRICE` as the target variable.

### Features

| Feature   | Description                                         |
| --------- | --------------------------------------------------- |
| `CRIM`    | Per capita crime rate by town                       |
| `ZN`      | Proportion of residential land zoned for large lots |
| `INDUS`   | Proportion of non-retail business acres             |
| `CHAS`    | Whether the property tract bounds the Charles River |
| `NOX`     | Nitric oxide concentration                          |
| `RM`      | Average number of rooms per dwelling                |
| `AGE`     | Proportion of older owner-occupied units            |
| `DIS`     | Distance to Boston employment centres               |
| `RAD`     | Accessibility to radial highways                    |
| `TAX`     | Property-tax rate                                   |
| `PTRATIO` | Pupil-teacher ratio                                 |
| `B`       | Dataset-specific demographic variable               |
| `LSTAT`   | Percentage of lower-income population               |
| `PRICE`   | Median value of owner-occupied homes in $1000s      |

---

## 🔎 1. Exploring the Data

I first loaded the dataset using Pandas and checked its basic structure.

```python
data = pd.read_csv('boston.csv', index_col=0)
```

I checked:

* Shape of the dataset
* Number of rows and columns
* Column names
* Duplicate values
* Missing values

```python
data.shape
data.columns
data.duplicated().sum()
data.isna().any()
```

I also used:

```python
data.describe()
```

to understand the minimum, maximum, mean and other basic statistics of the numerical features.

---

## 🧹 2. Data Cleaning

Before building the model, I checked whether the dataset contained:

* Missing values
* Duplicate rows
* Unexpected data issues

The dataset did not contain missing values, and I checked for duplicate records before continuing with the analysis.

This helped me understand why checking and preparing data is an important step before Machine Learning.

---

## 📈 3. Data Visualization

I used **Seaborn, Matplotlib and Plotly** to understand the data visually.

Some of the variables I explored were:

### House Prices

I plotted the distribution of `PRICE` to understand how home values were distributed.

### Number of Rooms

I explored `RM` to see how the number of rooms was distributed.

### Distance from Employment Centres

I visualized `DIS` to understand the distribution of distances from employment centres.

### Highway Accessibility

I explored `RAD` to understand how properties were distributed according to highway accessibility.

### Charles River

I used a Plotly bar chart to compare properties located next to the Charles River with properties that were not.

---

## 🔗 4. Understanding Relationships

I used Seaborn's `pairplot()` and `jointplot()` to investigate relationships between different variables.

Some relationships I explored:

* `NOX` vs `DIS`
* `INDUS` vs `NOX`
* `LSTAT` vs `RM`
* `LSTAT` vs `PRICE`
* `RM` vs `PRICE`

For example, I explored how the number of rooms relates to house prices and how the percentage of lower-income population relates to house prices.

This helped me understand that visualization can be useful before choosing and training a model.

---

## ✂️ 5. Training and Testing Data

I separated the target variable from the other features.

```python
target = data['PRICE']
features = data.drop('PRICE', axis=1)
```

Then I split the data into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=10
)
```

The training data was used to teach the model, while the test data was kept separate to check how the model performs on data it did not train on.

---

## 🤖 6. Multivariable Linear Regression

I used Scikit-learn's `LinearRegression` model.

```python
regr = LinearRegression()

regr.fit(X_train, y_train)
```

Because there are multiple input features, this is a **Multivariable Linear Regression** model.

The basic idea is:

```text
PRICE = intercept
        + rooms
        + pollution
        + distance
        + highway accessibility
        + ...
        + lower-income population
```

I also checked the model's R² score on the training data.

```python
rsquared = regr.score(X_train, y_train)
```

---

## 📋 7. Understanding Regression Coefficients

I examined the coefficients produced by the model.

```python
regr_coef = pd.DataFrame(
    data=regr.coef_,
    index=X_train.columns,
    columns=['Coefficient']
)
```

This helped me understand how the model uses each feature while making its prediction.

For example, I looked at the coefficient for `RM` to understand the model's estimated change associated with an additional room, while keeping the other features in the model.

---

## 📉 8. Predictions and Residuals

I generated predictions using:

```python
predicted_vals = regr.predict(X_train)
```

Then calculated residuals:

```python
residuals = y_train - predicted_vals
```

In simple terms:

```text
Residual = Actual Price - Predicted Price
```

I visualized:

1. Actual prices vs predicted prices
2. Residuals vs predicted prices
3. Distribution of residuals

The residual plots helped me understand where the model's predictions were different from the actual values.

---

## 📊 9. Residual Distribution

I also checked the:

* Mean of residuals
* Skewness of residuals
* Distribution of residuals

```python
resid_mean = residuals.mean()
resid_skew = residuals.skew()
```

I used a histogram with KDE to visually inspect the residual distribution.

This was part of learning how to check whether a regression model is behaving reasonably.

---

## 🔄 10. Log Transformation

The original `PRICE` distribution was not perfectly symmetric, so I experimented with a **log transformation**.

```python
y_log = np.log(data['PRICE'])
```

I compared:

```text
Original PRICE
       vs
Log-transformed PRICE
```

and checked their skewness.

The purpose was to see whether transforming the target variable could make the regression fit more suitable for the data.

---

## 🤖 11. Linear Regression with Log Prices

I trained another Linear Regression model using the log-transformed prices.

```python
log_regr = LinearRegression()

log_regr.fit(X_train, log_y_train)
```

Then I compared the R² scores of:

* Original Linear Regression
* Linear Regression using Log Prices

I also compared their:

* Predictions
* Residuals
* Residual distributions
* Coefficients
* Test performance

---

## 🧪 12. Testing the Models

I evaluated both models on the test dataset.

```python
regr.score(X_test, y_test)

log_regr.score(X_test, log_y_test)
```

This was important because a model should not only perform well on the data it has already seen.

The test data gave me a better idea of how the models performed on unseen data.

---

## 🏡 13. Estimating a Property Price

Finally, I used the trained model to estimate the price of a property.

First, I created a property using average values for the features.

```python
average_vals = features.mean().values
```

Then I used the trained regression model to make a prediction.

Since the second model predicts the **log of the price**, I converted the prediction back to the original price scale using:

```python
np.exp(log_estimate) * 1000
```

---

## 🏠 Example Property

I also created a custom property with characteristics such as:

```python
next_to_river = True
nr_rooms = 8
students_per_classroom = 20
distance_to_town = 5
pollution = data.NOX.quantile(q=0.75)
amount_of_poverty = data.LSTAT.quantile(q=0.25)
```

I kept the remaining features at their average values and changed the selected characteristics.

The trained model was then used to estimate the property's value.

---

## 📁 Project Structure

```text
Boston-Housing-Price-Prediction/
│
├── boston.csv
├── Boston_Housing_Price_Prediction.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Boston-Housing-Price-Prediction.git
```

### 2. Open the project

Open the notebook:

```text
Boston_Housing_Price_Prediction.ipynb
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### 4. Run the notebook

Run the cells from top to bottom.

Make sure `boston.csv` is in the same folder as the notebook.

---

## 📚 What This Project Represents

This project was mainly created as a **Machine Learning learning project**.

It helped me practice the complete workflow:

```text
Load Data
    ↓
Understand Data
    ↓
Check & Clean Data
    ↓
Explore Data
    ↓
Visualize Relationships
    ↓
Split Train/Test Data
    ↓
Build Linear Regression
    ↓
Check Coefficients
    ↓
Analyse Residuals
    ↓
Transform Data
    ↓
Train Another Model
    ↓
Compare Performance
    ↓
Make a Prediction
```

---

## 🔮 Future Learning

As I continue learning Machine Learning, I can extend this project by exploring:

* Polynomial Regression
* Ridge Regression
* Lasso Regression
* Decision Trees
* Random Forest
* Gradient Boosting
* XGBoost
* Feature Scaling
* Cross-Validation
* More regression evaluation metrics

---

## 📌 Disclaimer

This project is for **learning and practice purposes**.

The dataset represents historical Boston housing data and should not be used to estimate current real-world property prices in Boston.

---

## 🙌 Learning Source

This project was completed while learning Python, Pandas, data visualization and Machine Learning concepts through a hands-on Boston housing price prediction exercise.

The project focuses on understanding **how data analysis and Linear Regression work**, rather than building a production-ready real estate valuation system.
