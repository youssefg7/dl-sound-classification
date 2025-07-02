import subprocess
import time
import os
from dotenv import load_dotenv
from pyngrok import ngrok, conf

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Set ngrok token
ngrok_token = os.getenv("NGROK_AUTHTOKEN")
if not ngrok_token:
    raise ValueError("NGROK_AUTHTOKEN not found in .env file.")

conf.get_default().auth_token = ngrok_token

# === CONFIG ===
MLFLOW_PORT = 5000
USE_AUTH = False
AUTH_CREDENTIALS = "mlflow:1234"
# ==============

def start_mlflow_ui():
    print(f"▶ Starting MLflow UI on http://localhost:{MLFLOW_PORT}")
    return subprocess.Popen(["mlflow", "ui", "--port", str(MLFLOW_PORT)], stdout=subprocess.DEVNULL)

def expose_with_ngrok():
    print("Exposing MLflow UI with ngrok...")
    if USE_AUTH:
        url = ngrok.connect(MLFLOW_PORT, auth=AUTH_CREDENTIALS)
    else:
        url = ngrok.connect(MLFLOW_PORT)
    print(f"🔗 Public MLflow URL: {url}")
    return url

if __name__ == "__main__":
    try:
        mlflow_proc = start_mlflow_ui()
        time.sleep(3)  # Let MLflow initialize
        public_url = expose_with_ngrok()

        print("\nMLflow is live!")
        print(f"Visit: {public_url}")
        print("Press Ctrl+C to stop.")

        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n🧹 Stopping...")
    finally:
        ngrok.kill()
        mlflow_proc.terminate()