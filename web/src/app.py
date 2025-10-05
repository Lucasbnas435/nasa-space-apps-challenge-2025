import cloudpickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Carrega pipeline
with open("./models/lightgbm_pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

# Lista de features que queremos mostrar no formulário
feature_labels = {
    "koi_tce_delivname": "Delivery Name",
    "koi_period": "Orbital Period (days)",
    "koi_period_err1": "Period Error +",
    "koi_period_err2": "Period Error -",
    "koi_time0bk_err2": "Transit Epoch Error -",
    "koi_duration_err1": "Transit Duration Error +",
    "koi_duration_err2": "Transit Duration Error -",
    "koi_depth": "Transit Depth (ppm)",
    "koi_prad": "Planet Radius (Earth radii)",
    "koi_prad_err1": "Planet Radius Error +",
    "koi_prad_err2": "Planet Radius Error -",
    "koi_teq": "Equilibrium Temperature (K)",
    "koi_insol": "Insolation (Earth flux)",
    "koi_insol_err1": "Insolation Error +",
    "koi_insol_err2": "Insolation Error -",
    "koi_model_snr": "Model Signal-to-Noise Ratio",
    "koi_steff_err1": "Stellar Temperature Error +",
    "koi_steff_err2": "Stellar Temperature Error -",
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Pega os valores do formulário pelos nomes técnicos
        data = {
            tech_name: request.form[tech_name]
            for tech_name in feature_labels.keys()
        }
        X_new = pd.DataFrame([data])
        prediction = pipeline.predict(X_new)[0]
    return render_template(
        "home.html", feature_labels=feature_labels, prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
