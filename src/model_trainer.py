from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, recall_score
import pandas as pd
from sklearn.utils import resample

class ModelTrainer:
    def __init__(self, model=None):
        self.model = model if model else RandomForestClassifier(random_state=42)

    def balance_classes(self, X, y, method='oversample'):
        """Balancea las clases mediante sobremuestreo o submuestreo."""
        data = pd.concat([X, y], axis=1)
        mayoritaria = data[data['clase'] == 0]
        minoritaria = data[data['clase'] == 1]

        if method == 'oversample':  # Sobremuestreo
            minoritaria = resample(minoritaria, replace=True, n_samples=len(mayoritaria), random_state=42)
        elif method == 'undersample':  # Submuestreo
            mayoritaria = resample(mayoritaria, replace=False, n_samples=len(minoritaria), random_state=42)

        balanceado = pd.concat([mayoritaria, minoritaria])
        X_bal = balanceado.drop('clase', axis=1)
        y_bal = balanceado['clase']
        print(f"Clases balanceadas: {y_bal.value_counts().to_dict()}")
        return X_bal, y_bal

    def train_and_validate(self, X_train, y_train, X_val, y_val):
        """Entrena y valida el modelo utilizando un holdout."""
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        print("Reporte de clasificación:")
        print(classification_report(y_val, y_pred))
        print(f"F1-score para la clase 1: {f1_score(y_val, y_pred, pos_label=1):.2f}")
        print(f"Recall para la clase 1: {recall_score(y_val, y_pred, pos_label=1):.2f}")

    def analyze_feature_importance(self, X):
        """Muestra la importancia de las variables."""
        importances = self.model.feature_importances_
        importancia_df = pd.DataFrame({'Variable': X.columns, 'Importancia': importances})
        importancia_df = importancia_df.sort_values(by='Importancia', ascending=False)
        print("Variables más importantes:")
        print(importancia_df.head(10))
