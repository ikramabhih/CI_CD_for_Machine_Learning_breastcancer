import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Charger les données
df = pd.read_csv("Data/data.csv")

# Supprimer la dernière colonne inutile
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

# Mélanger les données
df = df.sample(frac=1, random_state=125)

# Séparer features et target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# Identifier les colonnes numériques et catégorielles
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

# Préprocessing
preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        ("cat", OrdinalEncoder(), cat_cols),
    ]
)

# Pipeline complet
pipe = Pipeline(
    [
        ("preprocessing", preprocess),
        ("model", RandomForestClassifier(n_estimators=10, random_state=125)),
    ]
)

# Entraîner le modèle
pipe.fit(X_train, y_train)

# Évaluation du modèle
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Créer dossiers Results et Model si nécessaire
os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)

# Écrire les métriques dans un fichier
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

# Sauvegarder le pipeline avec skops
sio.dump(pipe, "Model/breast_cancer_pipeline.skops")
