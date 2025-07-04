
#loding the data form here
import pandas as pd

df = pd.read_csv('data/diabetes.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())
 # preprocessing the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Outcome', axis=1)   # Features
y = df['Outcome']                # Target (1: diabetes, 0: healthy)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# training the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# save the model
import joblib
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')




