import pandas as pd
import sqlite3
import logging
import os
from datetime import datetime

class DataEnricher:
    def __init__(self, db_path=None):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(self.project_root, "src", "proyecto", "static", "data")
        self.models_dir = os.path.join(self.project_root, "src", "proyecto", "static", "models")

        if db_path is None:
            self.db_path = os.path.join(self.data_dir, "historical.db")
        else:
            self.db_path = db_path
        self.enriched_db_path = os.path.join(self.data_dir, "enriched_historical.db")

        self.logger = logging.getLogger('DataEnricher')
        self._setup_logger()
        print(f"DEBUG: Ruta de la base de datos de origen (Enricher): {self.db_path}")
        print(f"DEBUG: Ruta de la base de datos de destino (Enricher): {self.enriched_db_path}")
        print(f"DEBUG: Ruta del log (Enricher): {os.path.join(self.models_dir, 'enricher.log')}")

    def _setup_logger(self):
        os.makedirs(self.models_dir, exist_ok=True)
        log_filepath = os.path.join(self.models_dir, 'enricher.log')
        if self.logger.handlers:
            self.logger.handlers.clear()
        handler = logging.FileHandler(log_filepath, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def load_data(self):
        try:
            if not os.path.exists(self.db_path):
                self.logger.error(f"⚠ Archivo de base de datos 'historical.db' no encontrado en: {self.db_path}. Ejecute 'collector.py' primero.")
                print(f"⚠ Archivo de base de datos 'historical.db' no encontrado. Ejecute 'collector.py' primero.")
                return pd.DataFrame()

            conn = sqlite3.connect(self.db_path)
            query = "SELECT date, open, high, low, close, volume FROM historical"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                self.logger.warning("⚠ No se encontraron datos en la base de datos histórica para enriquecer.")
                print("⚠ No se encontraron datos en la base de datos histórica para enriquecer.")
                return pd.DataFrame()

            self.logger.info(f"✅ {len(df)} registros cargados desde la base de datos histórica.")
            print(f"✅ {len(df)} registros cargados desde la base de datos histórica.")
            return df
        except Exception as e:
            self.logger.error(f"⚠ Error al cargar datos desde la base de datos histórica ({self.db_path}): {e}")
            print(f"⚠ Error al cargar datos desde la base de datos histórica: {e}")
            return pd.DataFrame()

    def enrich_data(self, df):
        if df.empty:
            self.logger.warning("No hay datos para enriquecer.")
            return pd.DataFrame()

        try:
            # 1. Función para intentar convertir la fecha con diferentes formatos
            def try_parse_date(date_str):
                formats = ["%B %d, %Y", "%d-%m-%Y", "%Y-%m-%d", "%b %d, %Y", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S.%fZ"]
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        pass
                return None

            # 2. Convertir la columna 'date' usando la función
            df['date'] = df['date'].apply(try_parse_date)

            # 3. Eliminar filas donde la conversión de fecha falló
            df.dropna(subset=['date'], inplace=True)
            if df.empty:
                self.logger.error("⚠ No se pudieron convertir las fechas al formato datetime.")
                print("⚠ No se pudieron convertir las fechas al formato datetime.")
                return pd.DataFrame()

            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            self.logger.info("✅ Datos enriquecidos con variables temporales (año, mes, día, día de la semana, trimestre).")
            return df
        except Exception as e:
            self.logger.error(f"⚠ Error al enriquecer los datos: {e}")
            print(f"⚠ Error al enriquecer los datos: {e}")
            return pd.DataFrame()

    def save_enriched_data(self, df):
        if df.empty:
            self.logger.warning("No hay datos enriquecidos para guardar.")
            print("⚠ No hay datos enriquecidos para guardar.")
            return
        try:
            conn = sqlite3.connect(self.enriched_db_path)
            df.to_sql('enriched_historical', conn, if_exists='replace', index=False)
            conn.close()
            self.logger.info(f"✅ Datos enriquecidos guardados en: {self.enriched_db_path}")
            print(f"✅ Datos enriquecidos guardados en: {self.enriched_db_path}")
        except Exception as e:
            self.logger.error(f"⚠ Error al guardar los datos enriquecidos en {self.enriched_db_path}: {e}")
            print(f"⚠ Error al guardar los datos enriquecidos en {self.enriched_db_path}: {e}")

if __name__ == "__main__":
    enricher = DataEnricher()
    data = enricher.load_data()
    if not data.empty:
        enriched_data = enricher.enrich_data(data)
        if not enriched_data.empty:
            enricher.save_enriched_data(enriched_data)

