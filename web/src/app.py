from flask import Flask, render_template, request
from src.config.features import categorical_features, feature_info
from src.models.ml_pipeline_handler import MlPipelineHandler

app = Flask(__name__)

model = MlPipelineHandler(
    model_file_name="lightgbm_pipeline.pkl",
    categorical_features=categorical_features,
)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        form_data = request.form.to_dict()
        print(f"Form data: {form_data}")
        prediction = model.predict(form_data)

    return render_template(
        "home.html", feature_info=feature_info, prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
