import pandas as pd
import sqlite3
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

class DataModeller:
    def __init__(self):
        # --- CÁLCULO DE LA RUTA RAÍZ DEL PROYECTO (ProyectoIntegrado5/) ---
        # Este archivo (modeller.py) está en: .../ProyectoIntegrado5/src/modeller.py
        # Para llegar a la carpeta "ProyectoIntegrado5" desde aquí, solo necesitamos subir 1 nivel:
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # --- RUTAS A LAS CARPETAS 'static/data' y 'static/models' ---
        # Desde 'project_root' (ProyectoIntegrado5), construimos la ruta completa.
        # Ejemplo de ruta final: .../ProyectoIntegrado5/src/proyecto/static/models
        self.data_dir = os.path.join(self.project_root, "src", "proyecto", "static", "data")
        self.models_dir = os.path.join(self.project_root, "src", "proyecto", "static", "models")

        self.enriched_db_path = os.path.join(self.data_dir, "enriched_historical.db")
        self.model_path = os.path.join(self.models_dir, "model.pkl")

        self.logger = logging.getLogger('DataModeller')
        self._setup_logger()
        print(f"DEBUG: Ruta de la base de datos enriquecida (Modeller): {self.enriched_db_path}")
        print(f"DEBUG: Ruta para guardar el modelo (Modeller): {self.model_path}")
        print(f"DEBUG: Ruta del log (Modeller): {os.path.join(self.models_dir, 'modeller.log')}")


    def _setup_logger(self):
        # Asegura que la carpeta models_dir exista para guardar el log
        os.makedirs(self.models_dir, exist_ok=True)
        log_filepath = os.path.join(self.models_dir, 'modeller.log')

        # Eliminar handlers existentes para evitar duplicados en recargas de Streamlit
        if self.logger.handlers:
            self.logger.handlers.clear()

        handler = logging.FileHandler(log_filepath, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def load_enriched_data(self):
        try:
            if not os.path.exists(self.enriched_db_path):
                self.logger.error(f"⚠ Archivo de base de datos enriquecida no encontrado en: {self.enriched_db_path}. Ejecute 'enricher.py' primero.")
                print(f"⚠ Archivo de base de datos enriquecida no encontrado. Ejecute 'enricher.py' primero.")
                return pd.DataFrame() # Retorna DataFrame vacío

            conn = sqlite3.connect(self.enriched_db_path)
            query = "SELECT year, month, day, day_of_week, quarter, open, high, low, volume, close FROM enriched_historical"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                self.logger.warning("⚠ No se encontraron datos enriquecidos para el modelado.")
                print("⚠ No se encontraron datos enriquecidos para el modelado.")
                return pd.DataFrame()

            self.logger.info(f"✅ {len(df)} registros enriquecidos cargados para el modelado.")
            print(f"✅ {len(df)} registros enriquecidos cargados para el modelado.")
            return df
        except Exception as e:
            self.logger.error(f"⚠ Error al cargar datos enriquecidos ({self.enriched_db_path}): {e}")
            print(f"⚠ Error al cargar datos enriquecidos: {e}")
            return pd.DataFrame()

    def train_model(self, df):
        if df.empty: # Cambiado de 'is None' a 'empty'
            self.logger.warning("No hay datos para entrenar el modelo.")
            return None # Sigue retornando None para el modelo

        try:
            # df['date'] es type datetime64[ns], puede estar presente si se carga df_enriched completo
            df_cleaned = df.drop(columns=['date'], errors='ignore')

            features = ['year', 'month', 'day', 'day_of_week', 'quarter', 'open', 'high', 'low', 'volume']

            missing_features = [f for f in features if f not in df_cleaned.columns]
            if missing_features:
                self.logger.error(f"⚠ Faltan las siguientes columnas de features en los datos enriquecidos: {missing_features}")
                print(f"⚠ Faltan las siguientes columnas de features en los datos enriquecidos: {missing_features}")
                return None

            X = df_cleaned[features]
            y = df_cleaned['close'] # 'close' es la columna a predecir

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred) # Corrección: se elimina el parámetro squared
            rmse = rmse**0.5
            self.logger.info(f"✅ Modelo entrenado. RMSE en el conjunto de prueba: {rmse:.2f}")
            print(f"✅ Modelo entrenado. RMSE en el conjunto de prueba: {rmse:.2f}")
            return model
        except Exception as e:
            self.logger.error(f"⚠ Error al entrenar el modelo: {e}")
            print(f"⚠ Error al entrenar el modelo: {e}")
            return None

    def save_model(self, model):
        if model is None:
            self.logger.warning("No hay modelo para guardar.")
            return
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model, self.model_path)
            self.logger.info(f"✅ Modelo guardado en: {self.model_path}")
            print(f"✅ Modelo guardado en: {self.model_path}")
        except Exception as e:
            self.logger.error(f"⚠ Error al guardar el modelo: {e}")
            print(f"⚠ Error al guardar el modelo: {e}")

if __name__ == "__main__":
    modeller = DataModeller()
    enriched_data = modeller.load_enriched_data()
    if not enriched_data.empty: # Verifica si el DataFrame no está vacío
        model = modeller.train_model(enriched_data)
        if model is not None:
            modeller.save_model(model)