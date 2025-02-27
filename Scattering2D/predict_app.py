import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from model import FeatureExtractor, ImageClassifier
from utils import Utils

# ─── Configuración Inicial ────────────────────────────────────────────────
st.set_page_config(page_title="Clasificador de Imágenes", layout="wide", page_icon="📸")
st.title("📸 Clasificador de Imágenes con Scattering2D y PCA")
st.markdown(
    """
Esta aplicación utiliza un modelo de clasificación basado en Scattering2D y PCA para identificar imágenes.
Sube una o varias imágenes desde la barra lateral para obtener la predicción, la imagen más similar del dataset y
ver la proyección en 2D de las imágenes actualmente cargadas.
    """
)

# ─── Inicializar Historial de Predicciones ──────────────────────────────────
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []

# ─── BARRA LATERAL ───────────────────────────────────────────────────────────
st.sidebar.header("Carga de Imágenes")
uploaded_files = st.sidebar.file_uploader(
    "📤 Selecciona una o más imágenes",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# ─── CARGAR MODELO ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    feature_extractor = FeatureExtractor()
    classifier = ImageClassifier(feature_extractor)
    # Entrenar el modelo en modo preentrenado (no requiere conjunto de datos de entrenamiento)
    classifier.fit(df_train=None, is_pretrain=True)
    return classifier

classifier = load_model()

# ─── PROCESAMIENTO Y PREDICCIÓN DEL BATCH ACTUAL ───────────────────────────
current_predictions = []
if uploaded_files is not None and len(uploaded_files) > 0:
    progress_bar = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        with st.spinner(f"Procesando imagen {idx} de {len(uploaded_files)}..."):
            # Guardar la imagen en un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            # Leer la imagen utilizando la función de Utils
            image = Utils.read_images(temp_path)

            # Realizar la predicción
            predicted_label, best_match_path, min_distance = classifier.predict(temp_path, is_pretrain=True)
            os.remove(temp_path)

            # Guardar el resultado en el batch actual y en el historial persistente
            result = {
                "image": image,
                "predicted_label": predicted_label,
                "min_distance": min_distance,
                "best_match_path": best_match_path
            }
            current_predictions.append(result)
            st.session_state.predictions_history.append(result)

        progress_bar.progress(idx / len(uploaded_files))

    # ─── GALERÍA DE PREDICCIONES DEL BATCH ACTUAL ──────────────────────────────
    st.markdown("## Galería de Predicciones (Batch Actual)")
    for i, result in enumerate(current_predictions, start=1):
        st.markdown(f"**Imagen {i} - Predicción:** `{result['predicted_label']}` | **Distancia:** `{result['min_distance']:.4f}`")
        cols = st.columns(2)
        with cols[0]:
            st.image(result["image"], caption="Imagen Ingresada", width=250)
        with cols[1]:
            if result["best_match_path"]:
                st.image(result["best_match_path"], caption="Imagen Similar", width=250)
            else:
                st.info("No se encontró imagen similar.")
        st.markdown("---")

    # ─── VISUALIZACIÓN PCA BASADA EN EL BATCH ACTUAL ──────────────────────────
    st.markdown("## 🔎 Visualización PCA en 2D (Predicciones del Batch Actual)")
    with st.spinner("Generando visualización PCA..."):
        # Datos del dataset de entrenamiento (se asume que classifier.df_features tiene las componentes PCA)
        features = classifier.df_features.iloc[:, 2:4].values
        labels = classifier.df_features["label"].values
        pca_df = pd.DataFrame(features, columns=["PC1", "PC2"])
        pca_df["label"] = labels

        fig, ax = plt.subplots(figsize=(10, 6))
        # Graficar el dataset
        categories = pca_df["label"].unique()
        cmap = plt.get_cmap("tab10")
        for i, category in enumerate(categories):
            subset = pca_df[pca_df["label"] == category]
            ax.scatter(subset["PC1"], subset["PC2"], label=f"Dataset - {category}", alpha=0.5, color=cmap(i))
        
        # Graficar las predicciones del batch actual
        for i, result in enumerate(current_predictions, start=1):
            input_features = classifier.feature_extractor.transform(
                np.array([result["image"].astype(np.float32) / 255.0]),
                is_pretrain=True
            )
            ax.scatter(input_features[0, 0], input_features[0, 1], color="red", marker="*", s=200,
                       label=f"Predicción {i}: {result['predicted_label']}")
        
        ax.set_title("PCA - Representación 2D del dataset y predicciones actuales")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)
        # Eliminar etiquetas duplicadas en la leyenda
        handles, leg_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(leg_labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        st.pyplot(fig)

    st.markdown("🔴 **Las estrellas rojas representan las imágenes ingresadas en el batch actual y su ubicación en el espacio PCA.**")
else:
    st.info("Por favor, sube una o más imágenes desde la barra lateral para comenzar.")
    st.markdown("## 🔎 Visualización PCA en 2D")
    # Si no hay imágenes en el uploader, se muestra sólo el dataset de entrenamiento
    with st.spinner("Generando visualización PCA..."):
        features = classifier.df_features.iloc[:, 2:4].values
        labels = classifier.df_features["label"].values
        pca_df = pd.DataFrame(features, columns=["PC1", "PC2"])
        pca_df["label"] = labels

        fig, ax = plt.subplots(figsize=(10, 6))
        categories = pca_df["label"].unique()
        cmap = plt.get_cmap("tab10")
        for i, category in enumerate(categories):
            subset = pca_df[pca_df["label"] == category]
            ax.scatter(subset["PC1"], subset["PC2"], label=f"Dataset - {category}", alpha=0.5, color=cmap(i))
        
        ax.set_title("PCA - Representación 2D del dataset")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# ─── HISTORIAL DE PREDICCIONES (PERSISTENTE) ──────────────────────────────────
st.markdown("## Historial de Predicciones (Sesión)")
if st.session_state.predictions_history:
    for idx, entry in enumerate(st.session_state.predictions_history, start=1):
        st.markdown(f"**{idx}. Predicción:** `{entry['predicted_label']}` | **Distancia:** `{entry['min_distance']:.4f}`")
        cols = st.columns(2)
        with cols[0]:
            st.image(entry["image"], caption="Imagen Ingresada", width=150)
        with cols[1]:
            if entry["best_match_path"]:
                st.image(entry["best_match_path"], caption="Imagen Similar", width=150)
            else:
                st.info("No se encontró imagen similar.")
        st.markdown("---")
else:
    st.info("No hay predicciones previas en el historial.")
