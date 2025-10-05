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
            input_data = {
                col: [request.form[col]]
                for col in pipeline.named_steps["preprocessor"].transformers_[
                    1
                ][2]
            }
            df = pd.DataFrame(input_data)

            # --- Fazer predição ---
            prediction = pipeline.predict(df)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
