import cv2
import numpy as np
import os
import json
from skimage.measure import regionprops, label
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from imblearn.over_sampling import RandomOverSampler

# Define las rutas a los directorios que contienen tus imágenes y etiquetas
train_base_dir = '/home/ryuking/Documentos/practica laboral/HUVEC-DB2/train'
test_base_dir = '/home/ryuking/Documentos/practica laboral/HUVEC-DB2/test'
train_labels_file = '/home/ryuking/Documentos/practica laboral/HUVEC-DB2/train/labels.json'
test_labels_file = '/home/ryuking/Documentos/practica laboral/HUVEC-DB2/test/labels.json'

# Subdirectorios que contienen las imágenes
subdirs = ['circular', 'elongated', 'other']

# Función para binarizar una imagen
def binarize_image(image, threshold=127):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# Función para extraer características de una imagen binarizada
def extract_features(binary_image, num_features=11):
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    
    features = []
    for region in regions:
        perimeter = region.perimeter
        area = region.area
        eccentricity = region.eccentricity
        equivalent_diameter = region.equivalent_diameter
        major_axis_length = region.major_axis_length
        minor_axis_length = region.minor_axis_length
        solidity = region.solidity
        extent = region.extent
        orientation = region.orientation
        
        # Verificar si perimeter es cero para evitar división por cero
        if perimeter == 0:
            csf = 0  # O puedes usar np.inf o cualquier otro valor predeterminado
        else:
            csf = (4 * np.pi * area) / (perimeter ** 2)  # Circular Shape Factor
        
        # Verificar si minor_axis_length es cero para evitar división por cero
        if minor_axis_length == 0:
            esf = 0  # O puedes usar np.inf o cualquier otro valor predeterminado
        else:
            esf = major_axis_length / minor_axis_length  # Elongation Shape Factor
        
        features.append([
            perimeter, area, eccentricity, equivalent_diameter,
            major_axis_length, minor_axis_length, solidity, extent,
            orientation, csf, esf
        ])
    
    # Asegurarse de que todas las características tengan la misma longitud
    if len(features) > 0:
        features = np.array(features).flatten()
        if len(features) < num_features:
            features = np.pad(features, (0, num_features - len(features)), 'constant')
        elif len(features) > num_features:
            features = features[:num_features]
    else:
        features = np.zeros(num_features)
    
    return features

# Función para cargar imágenes y etiquetas desde un directorio base y un archivo JSON
def load_images_and_labels(base_dir, labels_file, subdirs, num_features=11):
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    images = []
    image_labels = []
    
    for subdir in subdirs:
        image_dir = os.path.join(base_dir, subdir)
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = binarize_image(img)  # Binarizar la imagen
                features = extract_features(img, num_features=num_features)  # Extraer características
                images.append(features)
                # Obtiene la etiqueta correspondiente desde el archivo JSON
                file_key = os.path.splitext(filename)[0]  # Elimina la extensión del archivo
                if file_key in labels:
                    label = labels[file_key]
                    image_labels.append(label)
    
    print(f"Se cargaron {len(images)} imágenes y {len(image_labels)} etiquetas desde {base_dir}")
    return np.vstack(images), np.array(image_labels)

print("Cargando imágenes y etiquetas de entrenamiento...")
# Cargar las imágenes y etiquetas de entrenamiento
X_train, y_train = load_images_and_labels(train_base_dir, train_labels_file, subdirs)

# Verificar que los datos de entrenamiento no estén vacíos
if X_train.size == 0 or y_train.size == 0:
    raise ValueError("El conjunto de datos de entrenamiento está vacío. Verifica que las imágenes y etiquetas se hayan cargado correctamente.")

print("Sobremuestreando las clases minoritarias en los datos de entrenamiento...")
# Sobremuestrear las clases minoritarias en los datos de entrenamiento
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print("Cargando imágenes y etiquetas de prueba...")
# Cargar las imágenes y etiquetas de prueba
X_test, y_test = load_images_and_labels(test_base_dir, test_labels_file, subdirs)

# Verificar que los datos de prueba no estén vacíos
if X_test.size == 0 or y_test.size == 0:
    raise ValueError("El conjunto de datos de prueba está vacío. Verifica que las imágenes y etiquetas se hayan cargado correctamente.")

# Normalizar las características
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Guardar el scaler
joblib.dump(scaler, 'scaler.pkl')

# Convertir las etiquetas de texto a valores enteros
label_encoder = LabelEncoder()
y_train_resampled = label_encoder.fit_transform(y_train_resampled)
y_test = label_encoder.transform(y_test)

# Guardar el label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Definir los modelos individuales
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42)
svm_model = SVC(kernel='rbf', C=5, gamma=1/(2*(3.3**2)), probability=True, random_state=42)

# Entrenar los modelos individuales
print("Entrenando el modelo Gradient Boosting...")
gb_model.fit(X_train_resampled, y_train_resampled)

print("Entrenando el modelo Random Forest...")
rf_model.fit(X_train_resampled, y_train_resampled)

print("Entrenando el modelo SVM...")
svm_model.fit(X_train_resampled, y_train_resampled)

# Definir el Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('gb', gb_model),
    ('rf', rf_model),
    ('svm', svm_model)
], voting='soft')

print("Entrenando el Voting Classifier...")
voting_clf.fit(X_train_resampled, y_train_resampled)

print("Guardando el Voting Classifier entrenado...")
# Guardar el modelo entrenado
joblib.dump(voting_clf, 'VotingClassifier_best_model.pkl')

print("Prediciendo con el Voting Classifier...")
# Predecir las etiquetas en el conjunto de prueba
y_pred = voting_clf.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Voting Classifier Best Model Accuracy: {accuracy * 100:.2f}%')

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f'Voting Classifier Confusion Matrix:\n{cm}')
