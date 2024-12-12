import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, raw_file_path, processed_dir):
        self.raw_file_path = raw_file_path
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)  # Crear el directorio si no existe

    def load_data(self):
        """Carga el archivo Excel y añade la columna 'clase'."""
        data = pd.read_excel(self.raw_file_path)
        data['clase'] = 0
        data.loc[0:13, 'clase'] = 1  # Etiquetar filas 1-14 como roturas
        print("Columna 'clase' añadida al dataset.")
        return data

    def handle_missing_values(self, data):
        """Rellena valores NaN con la media de cada columna numérica."""
        columnas_numericas = data.select_dtypes(include=['float', 'int']).columns
        for col in columnas_numericas:
            data[col] = data[col].fillna(data[col].mean())  # Rellenar NaN con la media
        print("Valores faltantes rellenados con la media de cada columna.")
        return data

    def create_time_windows(self, data, k=3):
        """Crea ventanas de tiempo concatenando valores de K días consecutivos."""
        columnas_sensores = [col for col in data.columns if col.startswith('S')]
        ventanas = []
        for i in range(k - 1, len(data)):
            ventana = data.iloc[i - (k - 1):i + 1]
            valores = ventana[columnas_sensores].values.flatten()
            etiqueta = ventana['clase'].iloc[-1]  # Etiqueta del último día
            ventanas.append(list(valores) + [etiqueta])
        columnas = [f"{col}_dia_{j}" for j in range(1, k + 1) for col in columnas_sensores] + ['clase']
        return pd.DataFrame(ventanas, columns=columnas)

    def standardize_data(self, data):
        """Estandariza columnas numéricas utilizando StandardScaler."""
        columnas_sensores = [col for col in data.columns if col != 'clase']
        data[columnas_sensores] = data[columnas_sensores].apply(pd.to_numeric, errors='coerce')
        data[columnas_sensores] = data[columnas_sensores].fillna(0)  # Manejo de valores NaN
        scaler = StandardScaler()
        data[columnas_sensores] = scaler.fit_transform(data[columnas_sensores])
        print("Datos estandarizados correctamente.")
        return data
