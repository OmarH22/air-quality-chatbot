import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('updated_pollution_dataset.csv', sep=';')

# Prepare features and target
features = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
X = data[features]
y = data['Air Quality']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the model and encoder
joblib.dump(rf_model, 'air_quality_rf_model.joblib')
joblib.dump({'Air_Quality': le}, 'air_quality_encoders.joblib')

# Optional: Train and save Decision Tree (for documentation requirement)
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'air_quality_dt_model.joblib')

print("Models and encoders saved successfully.")