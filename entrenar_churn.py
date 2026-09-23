import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. Cargar la base de datos
df = pd.read_csv('dataset_streaming_churn.csv')

# 2. Separar características (X) y la variable a predecir (y)
X = df.drop(columns=['ID_Cliente', 'Churn'])
y = df['Churn']

# 3. Dividir los datos: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar Modelo 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# 5. Entrenar Modelo 2: XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

# 6. Mostrar resultados
print("🏆 RESULTADOS DEL TORNEO DE MODELOS 🏆")
print("-" * 45)
print(f"Precisión Random Forest: {rf_acc * 100:.2f}%")
print(f"Precisión XGBoost:       {xgb_acc * 100:.2f}%")
print("-" * 45)

# 7. Explicabilidad: Feature Importance
importancias = pd.DataFrame({
    'Variable': X.columns,
    'Peso_Porcentual': rf_model.feature_importances_ * 100
}).sort_values(by='Peso_Porcentual', ascending=False)

print("\n🔍 MOTIVOS PRINCIPALES DE FUGA (Feature Importance):")
for index, row in importancias.iterrows():
    print(f"- {row['Variable']}: {row['Peso_Porcentual']:.1f}%")