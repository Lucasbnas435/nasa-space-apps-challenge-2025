import cloudpickle
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

# Caminho para o arquivo .pkl que você salvou
with open("./models/lightgbm_pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # --- Ler dados do formulário ---
            # Supondo que cada coluna original seja um campo do form
            # 1️⃣ Pega o JSON enviado no body
            json_data = request.get_json()

            # 2️⃣ Converte a lista de exemplos em DataFrame
            X_new = pd.DataFrame([json_data])

            # 3️⃣ Passa o DataFrame inteiro para o pipeline
            prediction = pipeline.predict(X_new)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
