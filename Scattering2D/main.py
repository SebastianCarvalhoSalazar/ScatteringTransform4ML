from utils import Utils  # Importar la clase Utils
from model import FeatureExtractor  # Importar FeatureExtractor
from model import ImageClassifier  # Importar ImageClassifier
from sklearn.metrics import classification_report

def main():
    # 1️⃣ Extraer y procesar imágenes desde el ZIP
    df = Utils.extract_and_process_images(zip_path="imagedatabase.zip", output_dir="processed_data", 
                                          classes=["Sesameseeds", "Rice", "Pearlsugar", "Oatmeal", "Lentils"],
                                          imgs_folder="kyberge", train_ratio=0.8)

    # 2️⃣ Dividir datos en entrenamiento y prueba
    train_df = df[df['set'] == 'train'] # 80% para entrenamiento
    test_df =  df[df['set'] == 'test']  # 20% para prueba

    print("Ejemplo de datos de entrenamiento:")
    print(train_df.head())

    # 3️⃣ Inicializar FeatureExtractor y ImageClassifier
    feature_extractor = FeatureExtractor()
    classifier = ImageClassifier(feature_extractor)

    # 4️⃣ Entrenar el clasificador con imágenes de entrenamiento
    print("Entrenando el modelo...")
    classifier.fit(train_df)
    print("Entrenamiento finalizado.")

    # 5️⃣ Probar el clasificador con imágenes de prueba
    print("\nProbando el modelo con imágenes de prueba:")
    y_true = []
    y_pred = []
    for i, row in test_df.sample(frac=1).iterrows():
        image_path = row["image_path"]
        true_label = row["label"]
        
        # Predicción
        predicted_label, best_match_path, distance = classifier.predict(image_path)
        
        # Mostrar resultados
        print(f"Imagen: {image_path}")
        print(f"Etiqueta real: {true_label} - Predicción: {predicted_label} (Distancia: {distance:.4f})")
        y_true.append(true_label)
        y_pred.append(predicted_label)
        print("-" * 50)

    # 6. Metricas
    labels = test_df["label"].unique()
    report = classification_report(y_true, y_pred, target_names=labels)
    print("\nMetricas:")
    print(report)

# Ejecutar solo si el script se corre directamente
if __name__ == "__main__":
    main()
