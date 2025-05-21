import requests
import sqlite3
import csv
import logging
import os
import time
from datetime import datetime
from bs4 import BeautifulSoup

class DataCollector:
    def __init__(self, url_base):
        self.url_base = url_base
        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.static_dir = os.path.join(self.project_dir, "static")
        self.data_dir = os.path.join(self.static_dir, "data")
        self.models_dir = os.path.join(self.static_dir, "models")

        self.db_path = os.path.join(self.data_dir, "historical.db")
        self.csv_path = os.path.join(self.data_dir, "historical.csv")
        self.log_path = os.path.join(self.models_dir, "collector.log")

        self.logger = logging.getLogger('DataCollector')
        self.setup_logger()
        self.ensure_directories()

    def setup_logger(self):
        """Configura el logging para la aplicación."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def ensure_directories(self):
        """Asegura que las carpetas `data` y `models` existan antes de guardar los archivos."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def build_dynamic_url(self):
        """Genera dinámicamente la URL con fechas ajustadas a los últimos 5 años."""
        end_date = int(time.time())
        start_date = end_date - (5 * 365 * 24 * 60 * 60)

        url = f"{self.url_base}?period1={start_date}&period2={end_date}&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true"
        print(f"📌 Usando URL dinámica: {url}")
        return url

    def fetch_data(self):
        """Obtiene datos de la página web de Yahoo Finance."""
        url = self.build_dynamic_url()
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')

            if not table:
                self.logger.error("⚠ No se encontró la tabla en la página.")
                return None

            data = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    date = cols[0].text.strip()
                    open_price = cols[1].text.replace(',', '').strip()
                    high_price = cols[2].text.replace(',', '').strip()
                    low_price = cols[3].text.replace(',', '').strip()
                    close_price = cols[4].text.replace(',', '').strip()
                    volume = cols[5].text.replace(',', '').strip()

                    data.append({
                        'date': date,
                        'open': float(open_price),
                        'high': float(high_price),
                        'low': float(low_price),
                        'close': float(close_price),
                        'volume': int(volume) if volume.isdigit() else 0
                    })
            self.logger.info("✅ Datos obtenidos correctamente.")
            print(f"Datos obtenidos: {data}")
            return data
        else:
            self.logger.error(f"⚠ Error al obtener los datos, código: {response.status_code}")
            return None

    def save_to_db(self, data):
        """Elimina registros antiguos y guarda datos nuevos en SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ✅ Asegurar que la tabla `historical` existe
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical (
                date TEXT PRIMARY KEY, 
                open REAL, 
                high REAL, 
                low REAL, 
                close REAL, 
                volume INTEGER
            )
        ''')
        conn.commit()  # Confirmar la creación de la tabla

        # 🔴 Eliminar registros antiguos antes de insertar datos nuevos
        cursor.execute("DELETE FROM historical")

        # ✅ Insertar los nuevos datos
        for entry in data:
            print(f"📌 Insertando en BD: {entry['date']}")  # Debugging
            cursor.execute('''
                INSERT INTO historical (date, open, high, low, close, volume) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (entry['date'], entry['open'], entry['high'], entry['low'], entry['close'], entry['volume']))
        
        conn.commit()
        conn.close()
        self.logger.info("✅ Base de datos actualizada correctamente")
        print("✅ Guardado en base de datos con actualización")

    def save_to_csv(self, data):
        """Guarda los datos en CSV."""
        with open(self.csv_path, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Fecha", "Apertura", "Máximo", "Mínimo", "Cierre", "Volumen"])
            for entry in data:
                writer.writerow([entry['date'], entry['open'], entry['high'], entry['low'], entry['close'], entry['volume']])

        self.logger.info('✅ Datos guardados correctamente en CSV')
        print("Guardado en CSV")

    def update_data(self):
        """Proceso completo de actualización."""
        data = self.fetch_data()
        if data:
            print("Iniciando proceso de guardado/actualización en BD...")
            self.save_to_db(data)
            self.save_to_csv(data)
        else:
            self.logger.error("⚠ No se pudieron obtener datos, el proceso se detiene.")

if __name__ == "__main__":
    collector = DataCollector(url_base="https://finance.yahoo.com/quote/ETH-USD/history")
    collector.update_data()