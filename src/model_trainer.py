from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score
import pandas as pd
from sklearn.utils import resample


class ModelTrainer:
    """
    Clase para manejar múltiples modelos supervisados.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "RandomForest": RandomForestClassifier(random_state=self.random_state),
            "XGBoost": XGBClassifier(random_state=self.random_state),
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "LogisticRegression": LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        self.model = None  # Modelo actual

    def set_model(self, model_name):
        """
        Selecciona el modelo a utilizar.
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no está disponible. Modelos disponibles: {list(self.models.keys())}")
        self.model = self.models[model_name]
        print(f"Modelo seleccionado: {model_name}")

    def balance_classes(self, X, y, method='oversample'):
        """
        Balancea las clases mediante sobremuestreo o submuestreo.
        """
        data = pd.concat([X, y], axis=1)
        mayoritaria = data[data['clase'] == 0]
        minoritaria = data[data['clase'] == 1]

        if method == 'oversample':  # Sobremuestreo
            minoritaria = resample(minoritaria, replace=True, n_samples=len(mayoritaria), random_state=self.random_state)
        elif method == 'undersample':  # Submuestreo
            mayoritaria = resample(mayoritaria, replace=False, n_samples=len(minoritaria), random_state=self.random_state)

        balanceado = pd.concat([mayoritaria, minoritaria])
        X_bal = balanceado.drop('clase', axis=1)
        y_bal = balanceado['clase']
        print(f"Clases balanceadas: {y_bal.value_counts().to_dict()}")
        return X_bal, y_bal

    def train_and_validate(self, X_train, y_train, X_val, y_val):
        """
        Entrena y valida el modelo actual.
        """
        # Validar si el modelo ha sido definido
        if self.model is None:
            raise ValueError("Ningún modelo ha sido seleccionado. Usa 'set_model()' para elegir uno.")

        # Entrenar modelo
        self.model.fit(X_train, y_train)

        # Validar modelo
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, "predict_proba") else None

        f1 = f1_score(y_val, y_pred, pos_label=1)
        recall = recall_score(y_val, y_pred, pos_label=1)
        print("Reporte de clasificación:")
        print(classification_report(y_val, y_pred))
        print(f"F1-score: {f1:.2f}, Recall: {recall:.2f}")

        return y_pred, y_pred_proba

    def analyze_feature_importance(self, X):
        """
        Muestra la importancia de las variables (solo para modelos con feature_importances_).
        """
        if not hasattr(self.model, "feature_importances_"):
            print("El modelo actual no soporta importancia de variables.")
            return

        importances = self.model.feature_importances_
        importancia_df = pd.DataFrame({'Variable': X.columns, 'Importancia': importances})
        importancia_df = importancia_df.sort_values(by='Importancia', ascending=False)
        print("Variables más importantes:")
        print(importancia_df.head(10))
