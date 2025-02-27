import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import Utils  # Importar la clase Utils
from model import FeatureExtractor, ImageClassifier  # Importar FeatureExtractor e ImageClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# Configuración inicial de la app
st.set_page_config(page_title="Clasificador de Imágenes", layout="wide")
st.title("Clasificador de Imágenes")
st.markdown("Una aplicación moderna y profesional para procesar y clasificar imágenes.")

# ─── BARRA LATERAL ──────────────────────────────────────────────────────────────
st.sidebar.header("Configuración de la Aplicación")
zip_path = st.sidebar.text_input("Ruta del ZIP", value="imagedatabase.zip")
output_dir = st.sidebar.text_input("Directorio de salida", value="processed_data")
classes = st.sidebar.multiselect(
    "Selecciona las clases",
    options=["Sesameseeds", "Rice", "Pearlsugar", "Oatmeal", "Lentils",'aluminium_foil','corduroy','cracker','linen','sponge'],
    default=["Sesameseeds", "Rice", "Pearlsugar", "Oatmeal", "Lentils"]
)
imgs_folder = st.sidebar.selectbox(
    "Carpeta de imágenes",
    options=["kyberge", "kth_tips"],
)
train_ratio = st.sidebar.slider("Proporción para entrenamiento", min_value=0.5, max_value=0.9, value=0.8)

# ─── PROCESAMIENTO DE IMÁGENES ─────────────────────────────────────────────────────
st.header("1️⃣ Extracción y Procesamiento de Imágenes")
if st.button("Iniciar procesamiento"):
    with st.spinner("Extrayendo y procesando imágenes..."):
        df = Utils.extract_and_process_images(
            zip_path=zip_path, 
            output_dir=output_dir, 
            classes=classes, 
            imgs_folder=imgs_folder, 
            train_ratio=train_ratio
        )
    st.success("¡Procesamiento completado!")
    
    # Mostrar algunos datos de entrenamiento
    train_df = df[df['set'] == 'train']
    st.subheader("Ejemplo de datos de entrenamiento")
    st.dataframe(train_df.sample(5))

    # ─── ENTRENAMIENTO DEL MODELO ────────────────────────────────────────────────
    st.header("2️⃣ Entrenamiento del Modelo")
    with st.spinner("Inicializando el modelo y entrenando..."):
        feature_extractor = FeatureExtractor(pca_variance=np.min([len(train_df),200]))
        classifier = ImageClassifier(feature_extractor)
        classifier.fit(train_df)
    st.success("Entrenamiento finalizado.")

    # ─── PRUEBA DEL MODELO ─────────────────────────────────────────────────────────
    st.header("3️⃣ Prueba del Modelo")
    test_df = df[df['set'] == 'test']
    y_true = []
    y_pred = []
    resultados = []

    # Se muestra una barra de progreso para las predicciones
    progress_text = "Realizando predicciones..."
    progress_bar = st.progress(0)
    total = len(test_df)
    for idx, (_, row) in enumerate(test_df.sample(frac=1).iterrows(), start=1):
        image_path = row["image_path"]
        true_label = row["label"]
        predicted_label, best_match_path, distance = classifier.predict(image_path)
        y_true.append(true_label)
        y_pred.append(predicted_label)
        resultados.append({
            "Imagen": image_path,
            "Etiqueta Real": true_label,
            "Predicción": predicted_label,
            "Distancia": f"{distance:.4f}"
        })
        progress_bar.progress(idx/total)
    
    st.subheader("Resultados de Prueba")
    st.table(pd.DataFrame(resultados))

    # ─── MÉTRICAS ─────────────────────────────────────────────────────────────────
    st.header("4️⃣ Métricas del Modelo")
    labels = test_df["label"].unique()
    # Se genera el reporte en formato dict para visualizarlo en JSON interactivo
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    st.subheader("Reporte de Clasificación")
    st.json(report)
    
    with st.expander("Ver reporte en formato texto"):
        report_text = classification_report(y_true, y_pred, target_names=labels)
        st.text(report_text)

    # ─── MATRIZ DE CONFUSIÓN ───────────────────────────────────────────────────────
    st.header("5️⃣ Matriz de Confusión")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta Real")
    ax.set_title("Matriz de Confusión")
    st.pyplot(fig)