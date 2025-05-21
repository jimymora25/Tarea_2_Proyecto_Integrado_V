import streamlit as st
import pandas as pd
import sqlite3
import joblib
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import logging

class Dashboard:
    def __init__(self):
        # --- CÁLCULO DE LA RUTA RAÍZ DEL PROYECTO (ProyectoIntegrado5/) ---
        # Este archivo (dashboard.py) está en: .../ProyectoIntegrado5/src/dashboard.py
        # Para llegar a la carpeta "ProyectoIntegrado5" desde aquí, subimos 1 nivel:
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # --- RUTAS A LAS CARPETAS 'static/data' y 'static/models' ---
        # Desde 'project_root' (ProyectoIntegrado5), construimos la ruta completa.
        # Ejemplo de ruta final: .../ProyectoIntegrado5/src/proyecto/static/data
        self.data_dir = os.path.join(self.project_root, "src", "proyecto", "static", "data")
        self.models_dir = os.path.join(self.project_root, "src", "proyecto", "static", "models")

        self.historical_db_path = os.path.join(self.data_dir, "historical.db")
        self.enriched_db_path = os.path.join(self.data_dir, "enriched_historical.db")
        self.model_path = os.path.join(self.models_dir, "model.pkl")

        self.logger = logging.getLogger('Dashboard')
        self._setup_logger()
        print(f"DEBUG: Ruta de la base de datos histórica (Dashboard): {self.historical_db_path}")
        print(f"DEBUG: Ruta de la base de datos enriquecida (Dashboard): {self.enriched_db_path}")
        print(f"DEBUG: Ruta del modelo (Dashboard): {self.model_path}")
        print(f"DEBUG: Ruta del log (Dashboard): {os.path.join(self.models_dir, 'dashboard.log')}")

    def _setup_logger(self):
        # Asegura que la carpeta models_dir exista para guardar el log
        os.makedirs(self.models_dir, exist_ok=True)
        log_filepath = os.path.join(self.models_dir, 'dashboard.log')

        # Eliminar handlers existentes para evitar duplicados en recargas de Streamlit
        if self.logger.handlers:
            self.logger.handlers.clear()

        handler = logging.FileHandler(log_filepath, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @st.cache_data(ttl=3600)  # Cachea los datos por 1 hora
    def load_data(_self): # <--- ¡AQUÍ ESTÁ EL CAMBIO IMPORTANTE!
        try:
            if not os.path.exists(_self.historical_db_path):
                _self.logger.error(f"⚠ Archivo de base de datos histórica no encontrado en: {_self.historical_db_path}.")
                st.error(f"Error: Base de datos histórica no encontrada. Ejecute 'collector.py' primero.")
                return pd.DataFrame()

            conn = sqlite3.connect(_self.historical_db_path)
            query = "SELECT date, open, high, low, close, volume FROM historical ORDER BY date ASC"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                _self.logger.warning("⚠ No se encontraron datos históricos en la base de datos.")
                st.warning("No se encontraron datos históricos. Ejecute 'collector.py'.")
                return pd.DataFrame()

            df['date'] = pd.to_datetime(df['date'])
            _self.logger.info("✅ Datos históricos cargados para el dashboard.")
            return df
        except Exception as e:
            _self.logger.error(f"⚠ Error al cargar datos históricos para el dashboard desde {_self.historical_db_path}: {e}")
            st.error(f"Error al cargar datos históricos: {e}. Asegúrate que 'collector.py' se ejecutó correctamente y que la base de datos está en la ruta esperada.")
            return pd.DataFrame()

    @st.cache_data(ttl=3600) # Cachea los datos por 1 hora
    def load_enriched_data(_self): # <--- ¡AQUÍ ESTÁ EL CAMBIO IMPORTANTE!
        try:
            if not os.path.exists(_self.enriched_db_path):
                _self.logger.error(f"⚠ Archivo de base de datos enriquecida no encontrado en: {_self.enriched_db_path}.")
                st.error(f"Error: Base de datos enriquecida no encontrada. Ejecute 'enricher.py' primero.")
                return pd.DataFrame()

            conn = sqlite3.connect(_self.enriched_db_path)
            df_enriched = pd.read_sql_query("SELECT * FROM enriched_historical", conn)
            conn.close()

            if df_enriched.empty:
                _self.logger.warning("⚠ No se encontraron datos enriquecidos en la base de datos.")
                st.warning("No se encontraron datos enriquecidos. Ejecute 'enricher.py'.")
                return pd.DataFrame()

            df_enriched['date'] = pd.to_datetime(df_enriched['date'])
            _self.logger.info("✅ Datos enriquecidos cargados para el dashboard.")
            return df_enriched
        except Exception as e:
            _self.logger.error(f"⚠ Error al cargar datos enriquecidos para el dashboard desde {_self.enriched_db_path}: {e}")
            st.error(f"Error al cargar datos enriquecidos: {e}. Asegúrate que 'enricher.py' se ejecutó y creó la base de datos.")
            return pd.DataFrame()

    @st.cache_resource # Cachea el modelo, ya que es un objeto grande
    def load_model(_self): # <--- ¡AQUÍ ESTÁ EL CAMBIO IMPORTANTE!
        try:
            if not os.path.exists(_self.model_path):
                _self.logger.error(f"⚠ Archivo del modelo no encontrado en: {_self.model_path}.")
                st.error(f"Error: Archivo del modelo no encontrado. Ejecute 'modeller.py' primero.")
                return None

            model = joblib.load(_self.model_path)
            _self.logger.info("✅ Modelo cargado para el dashboard.")
            return model
        except Exception as e:
            _self.logger.error(f"⚠ Error al cargar el modelo para el dashboard desde {_self.model_path}: {e}")
            st.error(f"Error al cargar el modelo: {e}. Asegúrate que 'modeller.py' se ejecutó y guardó el modelo.")
            return None

    def predict_next_day_price(self, model, last_day_data): # Aquí se mantiene 'self' porque no está cacheada
        if model is None or last_day_data.empty:
            self.logger.warning("No hay modelo o datos para predecir el precio del siguiente día.")
            return None
        try:
            # Si last_day_data es un DataFrame con una sola fila, iloc[0] lo convierte en una Serie
            last_day_series = last_day_data.iloc[0] if isinstance(last_day_data, pd.DataFrame) else last_day_data

            last_date = pd.to_datetime(last_day_series['date'])
            next_date = last_date + timedelta(days=1)

            # Valores de ejemplo para Open, High, Low y Volume del día siguiente
            # Esto es una suposición para la predicción, ya que no tenemos datos reales del futuro.
            # Podrías usar un promedio, el último valor, o una estimación más sofisticada.
            # Aquí se usa el precio de cierre del último día como base.
            next_day_features = {
                'open': last_day_series['close'],
                'high': last_day_series['close'] * 1.02, # Una subida del 2% como ejemplo
                'low': last_day_series['close'] * 0.98,  # Una bajada del 2% como ejemplo
                'volume': last_day_series['volume'], # Mismo volumen que el último día
                'year': next_date.year,
                'month': next_date.month,
                'day': next_date.day,
                'day_of_week': next_date.dayofweek,
                'quarter': next_date.quarter
            }

            features_df = pd.DataFrame([next_day_features])

            # Asegúrate de que las columnas estén en el orden que el modelo espera
            model_features = ['year', 'month', 'day', 'day_of_week', 'quarter', 'open', 'high', 'low', 'volume']
            features_df = features_df[model_features]

            predicted_price = model.predict(features_df)
            self.logger.info(f"✅ Predicción para el día siguiente ({next_date.strftime('%Y-%m-%d')}): {predicted_price[0]:,.2f}")
            return predicted_price[0]
        except Exception as e:
            self.logger.error(f"⚠ Error al predecir el precio del día siguiente: {e}")
            st.error(f"Error al realizar la predicción: {e}")
            return None

    def run(self):
        st.set_page_config(layout="wide")
        st.title("📈 Dashboard de Predicción de Precios de Criptomonedas")

        df_historical = self.load_data()
        df_enriched = self.load_enriched_data()
        model = self.load_model() # Aquí se carga el modelo

        # --- INICIO DE LAS LÍNEAS PARA INSPECCIONAR EL MODELO ---
        st.subheader("Detalles del Modelo de Regresión Lineal (solo para depuración)")
        if model:
            try:
                st.write(f"Coeficientes: {model.coef_}")
                st.write(f"Intercepto: {model.intercept_}")
            except AttributeError:
                st.write("El objeto cargado no tiene atributos .coef_ o .intercept_ (puede ser un tipo de modelo diferente).")
        else:
            st.info("El modelo no se ha cargado.")
        # --- FIN DE LAS LÍNEAS PARA INSPECCIONAR EL MODELO ---

        st.header("Datos Históricos Recientes")
        if not df_historical.empty:
            st.dataframe(df_historical.tail())
        else:
            st.warning("No se encontraron datos históricos. Por favor, asegúrate de que 'collector.py' se ejecutó correctamente.")

        st.header("Gráfico Interactivo de Precios de Cierre")
        if not df_historical.empty:
            fig = px.line(df_historical, x='date', y='close', title='Precio de Cierre a lo Largo del Tiempo')
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cargando datos para el gráfico... (Si no aparecen, revisa la ejecución de 'collector.py')")

        st.header("Predicción del Precio de Cierre para el Día Siguiente")
        # Condición para la predicción: modelo cargado Y datos enriquecidos no vacíos
        if model is not None and not df_enriched.empty:
            # Asegúrate de ordenar los datos para obtener el último día correctamente
            last_day_data_for_prediction = df_enriched.sort_values('date', ascending=False).iloc[0]
            predicted_price = self.predict_next_day_price(model, last_day_data_for_prediction)

            if predicted_price is not None:
                next_date_for_display = last_day_data_for_prediction['date'].date() + timedelta(days=1)
                st.metric(label=f"Precio de Cierre Predicho para {next_date_for_display}", value=f"${predicted_price:,.2f}")
            else:
                st.warning("No se pudo generar la predicción. Revisa los logs para más detalles.")
        else:
            st.error("El modelo no está cargado o los datos enriquecidos son insuficientes para realizar una predicción. Por favor, asegúrate de haber ejecutado 'enricher.py' y 'modeller.py' en ese orden.")

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
