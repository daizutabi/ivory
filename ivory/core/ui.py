import subprocess


def run(tracking_uri):
    try:
        subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])
    except KeyboardInterrupt:
        pass
