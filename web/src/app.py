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
            json_data = request.get_json()
            X_new = pd.DataFrame([json_data])

            # --- Listar colunas que o pipeline espera mas que não estão no JSON ---
            expected_columns = (
                pipeline.named_steps["preprocessor"].transformers_[0][2]
                + pipeline.named_steps["preprocessor"].transformers_[1][2]
            )  # categóricas + numéricas
            missing_cols = [
                col for col in expected_columns if col not in X_new.columns
            ]

            # --- Preencher essas colunas com NaN (ou valor neutro) ---
            for col in missing_cols:
                X_new[col] = 0.0

            # --- Garantir a mesma ordem de colunas ---
            X_new = X_new[expected_columns]

            # --- Fazer a predição ---
            prediction = pipeline.predict(X_new)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
