import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# 1️⃣ Load Dataset
df = pd.read_csv('retail_sales_dataset.csv')

# 2️⃣ Create Synthetic Date Column (if it doesn't exist)
if 'Date' not in df.columns:
    n = len(df)
    df['Date'] = pd.to_datetime(np.random.choice(
        pd.date_range('2023-01-01', '2023-12-31'), n
    ))

# 3️⃣ Convert Date to datetime and extract features
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek   # 0=Monday
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# 4️⃣ Encode Gender
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

# 5️⃣ One-Hot Encode Product Category
df = pd.get_dummies(df, columns=['Product Category'])

# 6️⃣ Optional: Age Group Feature
df['Age_Group'] = pd.cut(df['Age'], bins=[0,18,30,45,60,100],
                         labels=['0-18','19-30','31-45','46-60','60+'])
df = pd.get_dummies(df, columns=['Age_Group'])

# 7️⃣ Optional: Quantity Group Feature
df['Quantity_Group'] = pd.cut(df['Quantity'], bins=[0,1,3,5,10,100],
                              labels=['1','2-3','4-5','6-10','10+'])
df = pd.get_dummies(df, columns=['Quantity_Group'])

# 8️⃣ Define Features and Target
# WARNING: Do NOT include 'Total Amount' or any direct calculation (Quantity*Price) in X
# Also drop 'Date' column since we've extracted features from it
X = df.drop(['Transaction ID', 'Customer ID', 'Total Amount', 'Date'], axis=1)
y = df['Total Amount']

# 9️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 10️⃣ Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11️⃣ Evaluate Model
y_pred = model.predict(X_test)
print("R² score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# 12️⃣ Save Model for Django Integration
with open('sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training complete and saved as 'sales_model.pkl'.")