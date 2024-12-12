from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split

def main():
    # Configuración de rutas
    RAW_FILE_PATH = 'data/raw/data_rotura.xlsx'
    PROCESSED_DIR = 'data/processed/'

    # Instanciar clases
    processor = DataProcessor(RAW_FILE_PATH, PROCESSED_DIR)
    trainer = ModelTrainer()

    # Procesar datos
    data = processor.load_data()
    data = processor.handle_missing_values(data)
    dataset_ventanas = processor.create_time_windows(data, k=3)
    dataset_ventanas = processor.standardize_data(dataset_ventanas)

    # Dividir datos
    X = dataset_ventanas.drop(columns=['clase'])
    y = dataset_ventanas['clase']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Balancear clases (sobremuestreo)
    X_train_bal, y_train_bal = trainer.balance_classes(X_train, y_train, method='oversample')

    # Entrenar y validar modelo
    trainer.train_and_validate(X_train_bal, y_train_bal, X_val, y_val)

    # Analizar importancia de variables
    trainer.analyze_feature_importance(X)

if __name__ == "__main__":
    main()
