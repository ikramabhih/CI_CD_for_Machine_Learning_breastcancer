import gradio as gr
import skops.io as sio
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Supprimer les warnings liés aux versions
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Types de confiance pour skops
trusted_types = [
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing.StandardScaler",
    "sklearn.compose.ColumnTransformer",
    "sklearn.preprocessing.OrdinalEncoder",
    "sklearn.impute.SimpleImputer",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "numpy.dtype",
]

# Charger le pipeline entraîné pour Breast Cancer
pipe = sio.load("./Model/bc_pipeline.skops", trusted=trusted_types)

# Fonction de prédiction
def predict_cancer(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness):
    """Predict breast cancer (Malignant/Benign) based on diagnostic features.

    Args:
        mean_radius (float): Mean radius of tumor
        mean_texture (float): Mean texture
        mean_perimeter (float): Mean perimeter
        mean_area (float): Mean area
        mean_smoothness (float): Mean smoothness

    Returns:
        str: Predicted cancer type
    """
    features = [[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]]
    predicted = pipe.predict(features)[0]
    return f"Predicted: {predicted}"

# Inputs pour l'interface Gradio
inputs = [
    gr.Slider(6, 30, step=0.1, label="Mean Radius"),
    gr.Slider(9, 40, step=0.1, label="Mean Texture"),
    gr.Slider(40, 190, step=0.1, label="Mean Perimeter"),
    gr.Slider(140, 2500, step=1, label="Mean Area"),
    gr.Slider(0.05, 0.2, step=0.001, label="Mean Smoothness"),
]

outputs = [gr.Label(num_top_classes=2)]

# Quelques exemples pour tester le modèle rapidement
examples = [
    [14.3, 20.0, 92.0, 600.0, 0.1],
    [17.5, 25.0, 115.0, 800.0, 0.15],
    [10.0, 15.0, 65.0, 350.0, 0.08],
]

# Titre et description de l'app
title = "Breast Cancer Classification"
description = "Enter diagnostic features to predict Malignant or Benign."
article = (
    "This app is part of a CI/CD MLOps workflow. "
    "It demonstrates automated training, evaluation, and deployment of a Random Forest model "
    "to Hugging Face Spaces using GitHub Actions."
)

# Lancer l'interface Gradio
gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
