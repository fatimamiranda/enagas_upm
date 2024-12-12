from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.report_generator import ReportGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, roc_auc_score
import pandas as pd


def process_data(raw_file_path, processed_dir):
    """
    Procesa los datos originales y genera el dataset para entrenamiento.
    """
    processor = DataProcessor(raw_file_path, processed_dir)
    data = processor.load_data()
    data = processor.handle_missing_values(data)
    dataset_ventanas = processor.create_time_windows(data, k=3)
    return processor.standardize_data(dataset_ventanas)


def train_models(X_train, X_val, y_train, y_val):
    """
    Entrena y valida múltiples modelos supervisados.
    """
    trainer = ModelTrainer()
    report_gen = ReportGenerator()
    modelos = ["RandomForest", "XGBoost", "DecisionTree", "LogisticRegression"]
    resultados = []

    for modelo in modelos:
        # Seleccionar modelo
        trainer.set_model(modelo)

        # Entrenar y validar
        y_pred, y_pred_proba = trainer.train_and_validate(X_train, y_train, X_val, y_val)

        # Generar métricas
        metrics = {
            "Modelo": modelo,
            "F1-score": f1_score(y_val, y_pred, pos_label=1),
            "Recall": recall_score(y_val, y_pred, pos_label=1),
            "AUC": roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None
        }
        resultados.append(metrics)

        # Generar gráficos e informes
        if modelo == "XGBoost":
            cm_path = report_gen.plot_confusion_matrix_manual(trainer.model, X_val, y_val, modelo, "reports")
        else:
            cm_path = report_gen.plot_confusion_matrix(trainer.model, X_val, y_val, modelo)
            roc_path = report_gen.plot_roc_curve(trainer.model, X_val, y_val, modelo) if y_pred_proba is not None else None
            report_gen.generate_pdf_report(modelo, classification_report(y_val, y_pred), cm_path, roc_path, metrics)

    return pd.DataFrame(resultados)


def compare_models(resultados_df):
    """
    Compara los modelos y guarda los resultados en un CSV y PDF.
    """
    resultados_df = resultados_df.sort_values(by="F1-score", ascending=False)
    resultados_df.to_csv("reports/comparacion_modelos.csv", index=False)
    print("Comparación de modelos guardada en 'reports/comparacion_modelos.csv'.")
    #convertir_pdf(resultados_df)

#crear metodo convertir pdf
def convertir_pdf(resultados_df):
    # Crear un informe en PDF consolidado
    pdf_path = "reports/comparacion_modelos.pdf"
    with open(pdf_path, "w") as f:
        f.write(resultados_df.to_string())
    print(f"Informe PDF guardado en: {pdf_path}")


def main():
    # Configuración de rutas
    RAW_FILE_PATH = 'data/raw/data_rotura.xlsx'
    PROCESSED_DIR = 'data/processed/'

    # Procesar datos
    dataset = process_data(RAW_FILE_PATH, PROCESSED_DIR)

    # Dividir datos
    X = dataset.drop(columns=['clase'])
    y = dataset['clase']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Balancear clases
    trainer = ModelTrainer()
    X_train_bal, y_train_bal = trainer.balance_classes(X_train, y_train, method='oversample')

    # Entrenar modelos y generar informes
    resultados_df = train_models(X_train_bal, X_val, y_train_bal, y_val)

    # Comparar modelos y guardar el informe consolidado
    compare_models(resultados_df)


if __name__ == "__main__":
    main()
