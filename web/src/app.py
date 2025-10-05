import cloudpickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Carrega pipeline
with open("./models/lightgbm_pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

feature_info = {
    "koi_tce_delivname": {
        "label": "TCE Delivery Name",
        "tooltip": "TCE delivery name corresponding to the TCE data federated to the KOI.",
    },
    "koi_period": {
        "label": "Orbital Period (days)",
        "tooltip": "The interval between consecutive planetary transits.",
    },
    "koi_period_err1": {
        "label": "Orbital Period Upper Uncertainty (days)",
        "tooltip": "The interval between consecutive planetary transits.",
    },
    "koi_period_err2": {
        "label": "Orbital Period Lower Uncertainty (days)",
        "tooltip": "The interval between consecutive planetary transits.",
    },
    "koi_time0bk_err2": {
        "label": "Transit Epoch Lower Uncertainty (BJD - 2,454,833.0)",
        "tooltip": "The time corresponding to the center of the first detected transit in Barycentric Julian Day (BJD) minus a constant offset of 2,454,833.0 days. The offset corresponds to 12:00 on Jan 1, 2009 UTC.",
    },
    "koi_duration_err1": {
        "label": "Transit Duration Upper Uncertainty (hours)",
        "tooltip": "The duration of the observed transits. Duration is measured from first contact between the planet and star until last contact. Contact times are typically computed from a best-fit model produced by a Mandel-Agol (2002) model fit to a multi-quarter Kepler light curve, assuming a linear orbital ephemeris.",
    },
    "koi_duration_err2": {
        "label": "Transit Duration Lower Uncertainty (hours)",
        "tooltip": "The duration of the observed transits. Duration is measured from first contact between the planet and star until last contact. Contact times are typically computed from a best-fit model produced by a Mandel-Agol (2002) model fit to a multi-quarter Kepler light curve, assuming a linear orbital ephemeris.",
    },
    "koi_depth": {
        "label": "Transit Depth (parts per million)",
        "tooltip": "The fraction of stellar flux lost at the minimum of the planetary transit. Transit depths are typically computed from a best-fit model produced by a Mandel-Agol (2002) model fit to a multi-quarter Kepler light curve, assuming a linear orbital ephemeris.",
    },
    "koi_prad": {
        "label": "Planetary Radius (Earth radii)",
        "tooltip": "The radius of the planet. Planetary radius is the product of the planet star radius ratio and the stellar radius.",
    },
    "koi_prad_err1": {
        "label": "Planetary Radius Upper Uncertainty (Earth radii)",
        "tooltip": "The radius of the planet. Planetary radius is the product of the planet star radius ratio and the stellar radius.",
    },
    "koi_prad_err2": {
        "label": "Planetary Radius Lower Uncertainty (Earth radii)",
        "tooltip": "The radius of the planet. Planetary radius is the product of the planet star radius ratio and the stellar radius.",
    },
    "koi_teq": {
        "label": "Equilibrium Temperature (Kelvin)",
        "tooltip": "Approximation for the temperature of the planet. The calculation of equilibrium temperature assumes a) thermodynamic equilibrium between the incident stellar flux and the radiated heat from the planet, b) a Bond albedo (the fraction of total power incident upon the planet scattered back into space) of 0.3, c) the planet and star are blackbodies, and d) the heat is evenly distributed between the day and night sides of the planet.",
    },
    "koi_insol": {
        "label": "Insolation Flux [Earth flux]",
        "tooltip": "Insolation flux is another way to give the equilibrium temperature. It depends on the stellar parameters (specifically the stellar radius and temperature), and on the semi-major axis of the planet. It's given in units relative to those measured for the Earth from the Sun.",
    },
    "koi_insol_err1": {
        "label": "Insolation Flux Upper Uncertainty [Earth flux]",
        "tooltip": "Insolation flux is another way to give the equilibrium temperature. It depends on the stellar parameters (specifically the stellar radius and temperature), and on the semi-major axis of the planet. It's given in units relative to those measured for the Earth from the Sun.",
    },
    "koi_insol_err2": {
        "label": "Insolation Flux Lower Uncertainty [Earth flux]",
        "tooltip": "Insolation flux is another way to give the equilibrium temperature. It depends on the stellar parameters (specifically the stellar radius and temperature), and on the semi-major axis of the planet. It's given in units relative to those measured for the Earth from the Sun.",
    },
    "koi_model_snr": {
        "label": "Transit Signal-to-Noise",
        "tooltip": "Transit depth normalized by the mean uncertainty in the flux during the transits.",
    },
    "koi_steff_err1": {
        "label": "Stellar Effective Temperature Upper Uncertainty (Kelvin)",
        "tooltip": "The photospheric temperature of the star.",
    },
    "koi_steff_err2": {
        "label": "Stellar Effective Temperature Lower Uncertainty (Kelvin)",
        "tooltip": "The photospheric temperature of the star.",
    },
}

# lista fixa de colunas categóricas
categorical_features = [
    "koi_tce_delivname",
    "koi_tce_delivname_q1_q16_tce",
    "koi_tce_delivname_q1_q17_dr24_tce",
    # adicione aqui outras categóricas se houver
]
expected_columns = (
    pipeline.named_steps["preprocessor"].transformers_[0][2]
    + pipeline.named_steps["preprocessor"].transformers_[1][2]
)  # categóricas + numéricas
# valores padrão
default_values = {
    col: ("unknown" if col in categorical_features else 0.0)
    for col in expected_columns
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        form_dict = request.form.to_dict()
        X_new = pd.DataFrame([form_dict])

        X_new = X_new.reindex(columns=expected_columns)

        # converte colunas numéricas para float
        for col in expected_columns:
            if col not in categorical_features:
                X_new[col] = pd.to_numeric(X_new[col], errors="coerce")

        # preenche NaNs com valores padrão
        X_new = X_new.fillna(value=default_values)

        prediction = pipeline.predict(X_new)[0]

    return render_template(
        "home.html", feature_info=feature_info, prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)
