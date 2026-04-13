from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Загружаем датасет
iris = load_iris()
X = iris.data
y = iris.target

# Делим данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучаем модель
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Сохраняем модель
joblib.dump(model, "model.joblib")

print("Model saved successfully as model.joblib")