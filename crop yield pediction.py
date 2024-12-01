import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\Public\Seven sem\crop yield prediction project\yield_df.csv")

# Data cleaning
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# Drop invalid rows
to_drop = df[df['average_rain_fall_mm_per_year'].apply(lambda x: isinstance(x, str))].index
df.drop(to_drop, inplace=True)

# Select relevant columns
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]

# Plot distribution of crop yield
plt.figure(figsize=(10, 6))
sns.histplot(df['hg/ha_yield'], kde=True)
plt.title('Distribution of Crop Yield')
plt.xlabel('Crop Yield (hg/ha)')
plt.ylabel('Frequency')
plt.show()

# Plot correlation heatmap excluding categorical columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Split data into features and target variable
x = df.drop(columns='hg/ha_yield')
y = df['hg/ha_yield']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Preprocessing
one = OneHotEncoder(drop='first')
rscaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehotencoder', one, [4, 5]),
        ('standardization', rscaler, [0, 1, 2, 3])
    ],
    remainder='passthrough'
)

# Fit the preprocessor only on training data
x_train_dummy = preprocessor.fit_transform(x_train)
x_test_dummy = preprocessor.transform(x_test)  # Transform test data with fitted preprocessor

# Initialize models
model = {
    'Lr': LinearRegression(),
    'lss': Lasso(),
    'Rid': Ridge(),
    'knn': KNeighborsRegressor(),
    'Dt': DecisionTreeRegressor(),
    'Rf': RandomForestRegressor(),
    'Br': BaggingRegressor(),
    'GBR': GradientBoostingRegressor()
}

# Train models and print results
results = []
for name, mod in model.items():
    mod.fit(x_train_dummy, y_train)
    y_pred = mod.predict(x_test_dummy)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, r2))
    print(f'{name} MSE: {mse:.2f} Score: {r2:.2f}')

# Plot model performance
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2'])
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='MSE', data=results_df)
plt.title('Model Performance (MSE)')
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.show()

# Example using RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train_dummy, y_train)

# Plot feature importance
importances = rf.feature_importances_
features = preprocessor.get_feature_names_out()
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(x_train_dummy.shape[1]), importances[indices], align='center')
plt.xticks(range(x_train_dummy.shape[1]), features[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Prediction function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    feature = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
    feature = preprocessor.transform(feature)  # Transform the input
    pred = rf.predict(feature).reshape(1, -1)
    return pred[0]

# Define inputs for prediction
Year = 2023
average_rain_fall_mm_per_year = 1000
pesticides_tonnes = 100
avg_temp = 25.87
Area = 'Albania'
Item = 'Wheat'

# Get and print result
result = prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)
print("Predicted Crop Yield:", result)  # Ensure output is shown
