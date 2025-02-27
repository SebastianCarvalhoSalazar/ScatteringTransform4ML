import os
import cv2
import zipfile
import random
import numpy as np
import pandas as pd

class Utils:

    @staticmethod
    def read_images(img_path):
        """
        Lee una imagen en escala de grises, la normaliza y la redimensiona a 200x200.

        Parámetros:
        - img_path (str): Ruta de la imagen.

        Retorna:
        - np.ndarray: Imagen procesada o None si la imagen no pudo ser cargada.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
        if img is None:
            return None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200, 200))   # Redimensionar a 200x200        

        return img

    @staticmethod
    def extract_and_process_images(zip_path="imagedatabase.zip", output_dir="processed_data",
                                   classes=["Sesameseeds", "Rice", "Pearlsugar", "Oatmeal", "Lentils"],
                                   imgs_folder="kyberge", train_ratio=0.8):
        """
        Extrae imágenes de un archivo ZIP, las convierte a escala de grises, las redimensiona a 200x200
        y las divide en conjuntos de entrenamiento y prueba.

        Parámetros:
        - zip_path: Ruta del archivo ZIP.
        - output_dir: Carpeta donde se guardarán las imágenes procesadas.
        - train_ratio: Proporción de imágenes que irán al conjunto de entrenamiento (el resto será test).

        Retorna:
        - DataFrame con las rutas de imágenes y sus etiquetas.
        """

        # Extraer ZIP
        extract_path = "extracted_images"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Definir rutas de salida
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Leer imágenes y procesarlas
        image_data = []
        base_path = os.path.join(extract_path, imgs_folder)

        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            images = sorted(os.listdir(class_path))
            random.shuffle(images)  # Mezclar imágenes para train/test split

            split_index = int(len(images) * train_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]

            # Procesar imágenes
            for img_name in images:
                img_path = os.path.join(class_path, img_name)

                img = Utils.read_images(img_path)
                if img is None:
                    continue

                # Definir destino
                is_train = img_name in train_images
                dest_folder = train_dir if is_train else test_dir
                dest_class_folder = os.path.join(dest_folder, class_name)
                os.makedirs(dest_class_folder, exist_ok=True)

                # Guardar imagen procesada
                new_img_path = os.path.join(dest_class_folder, img_name)
                cv2.imwrite(new_img_path, img)

                # Guardar datos en DataFrame
                image_data.append({"image_path": new_img_path, "label": class_name, "set": "train" if is_train else "test"})

        # Crear DataFrame
        df = pd.DataFrame(image_data)

        return df

    @staticmethod
    def read_images_bytesio(img_path):
        """
        Lee una imagen desde una ruta de archivo o un objeto de Streamlit (BytesIO).
        Devuelve la imagen en escala de grises como un array de NumPy.
        """
        if isinstance(img_path, str):  # Si es una ruta de archivo
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:  # Si es un archivo subido (BytesIO)
            file_bytes = np.asarray(bytearray(img_path.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("No se pudo cargar la imagen. Verifica el archivo.")

        return img