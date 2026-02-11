import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/cleaned_ai3i2020.csv') 
model = joblib.load('models/random_forest_model.pkl') 
scaler = joblib.load('models/scaler.pkl')

X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Model Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('assets/confusion_matrix.png', dpi=300)
print("Generated assets/confusion_matrix.png")

plt.figure(figsize=(10, 6))
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance for Machine Failure', fontsize=16)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('assets/feature_importance.png', dpi=300)
print("Generated assets/feature_importance.png")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right")
plt.savefig('assets/roc_curve.png', dpi=300)
print("Generated assets/roc_curve.png")
