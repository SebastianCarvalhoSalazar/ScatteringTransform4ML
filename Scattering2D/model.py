import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from kymatio.sklearn import Scattering2D
import joblib
from utils import Utils

class FeatureExtractor:
    def __init__(self, scattering_model=None, pca_variance=200, shape=(200, 200), J=6, L=6):
        """
        Inicializa el extractor de características con Scattering2D, PCA y StandardScaler.
        """
        self.scattering_model = scattering_model or Scattering2D(J=J, shape=shape, L=L, max_order=2)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance)
        self.is_trained = False  # Bandera para verificar si se ha entrenado

    def compute_scattering2D(self, dataset=None, img=None):
        """
        Calcula la transformada Scattering2D del dataset o aplica scattering(img).
        """
        if dataset is not None:
            return self.scattering_model.transform(dataset)

        if img is not None:
            img = np.expand_dims(img, axis=0)
            return self.scattering_model(img)

        warnings.warn("Advertencia: Debe proporcionarse dataset o img.")
        return None        

    def fit(self, images):
        """
        Entrena el StandardScaler y PCA con el conjunto de imágenes dado.
        """
        if images is None or len(images) == 0:
            raise ValueError("Las imágenes de entrada están vacías.")

        scattering_features = self.compute_scattering2D(dataset=images)
        if scattering_features is None:
            raise ValueError("No se pudieron calcular las características de Scattering2D.")

        # Entrenamos el StandardScaler y PCA
        scaled_features = self.scaler.fit_transform(scattering_features)
        pca_features = self.pca.fit_transform(scaled_features)
        
        self.is_trained = True  # Marcamos que el modelo ha sido entrenado

        # Guardar los modelos entrenados
        joblib.dump(self.scaler, "./model/scaler.pkl")
        joblib.dump(self.pca, "./model/pca.pkl")

        return pca_features

    def transform(self, images, is_pretrain=False):
        """
        Transforma imágenes usando los modelos ya entrenados de Scattering2D, StandardScaler y PCA.
        """
        if is_pretrain:
            # Cargar los modelos entrenados
            self.scaler = joblib.load("./model/scaler.pkl")
            self.pca = joblib.load("./model/pca.pkl")

        if not self.is_trained and not is_pretrain:
            raise ValueError("FeatureExtractor no ha sido entrenado. Llama a `fit()` antes de `transform()`.")

        scattering_features = self.compute_scattering2D(dataset=images)
        if scattering_features is None:
            raise ValueError("No se pudieron calcular las características de Scattering2D.")
        

        # Aplicamos solo transform, sin volver a entrenar
        scaled_features = self.scaler.transform(scattering_features)
        pca_features = self.pca.transform(scaled_features)

        return pca_features



class ImageClassifier:
    def __init__(self, feature_extractor):
        """
        Inicializa el clasificador con un extractor de características.
        """
        self.feature_extractor = feature_extractor
        self.df_features = None

    def fit(self, df_train, is_pretrain=False):
        """
        Extrae y almacena características del conjunto de entrenamiento.
        """

        if is_pretrain:
            self.df_features = pd.read_csv("./model/train_features.csv", sep=";")
        else:
            test_imgs = []
            image_paths = []
            labels = []
            
            for img_path, label in zip(df_train["image_path"].values, df_train["label"].values):
                img = Utils.read_images(img_path)
                img = img.astype(np.float32) / 255.0
                if img is None:
                    continue
                test_imgs.append(img)
                image_paths.append(img_path)
                labels.append(label)
            
            if len(test_imgs) == 0:
                raise ValueError("No se cargaron imágenes válidas en el conjunto de entrenamiento.")

            test_imgs_arr = np.array(test_imgs)
            features = self.feature_extractor.fit(test_imgs_arr)  # Ahora llamamos a `fit()` para entrenar

            df_wst_features = pd.DataFrame(features)
            df_wst_features.insert(0, "image_path", image_paths)
            df_wst_features.insert(1, "label", labels)
            
            self.df_features = df_wst_features
            self.df_features.to_csv("./model/train_features.csv", sep=";", index=False)

    def predict(self, image_path, is_pretrain=False):
        """
        Predice la etiqueta de una imagen usando distancia coseno.
        """
        if is_pretrain:
            self.df_features = pd.read_csv("./model/train_features.csv", sep=";")

        if self.df_features is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a `fit()` antes de `predict()`.")
        
        img = Utils.read_images(image_path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

        # Extraer características usando `transform()`, NO `fit()`
        img = img.astype(np.float32) / 255.0
        img_features = self.feature_extractor.transform(np.array([img]), is_pretrain=is_pretrain)

        min_distance = float("inf")
        best_match_label = None
        
        for _, row in self.df_features.iterrows():
            db_features = np.array(row.iloc[2:])
            distance = cosine(img_features.flatten(), db_features)
            
            if distance < min_distance:
                min_distance = distance
                best_match_label = row["label"]
                best_match_path = row["image_path"]
        
        return best_match_label, best_match_path, min_distance
