# Documentación Técnica del Proyecto de Clasificación de Señas en LSC

## To view Tensorboard

```tensorboard --logdir logs/```

## 1. Introducción y Alcance

Este proyecto tiene como objetivo el reconocimiento de señas de la **Lengua de Señas Colombiana (LSC)** y su clasificación en cinco categorías que representan las palabras/frases:

1. **Café**  
2. **Aromática**  
3. **Con Gusto**  
4. **Qué necesita**  
5. **Hola**

Se cuenta con un **dataset** balanceado, compuesto por 1500 videos (300 por cada clase), capturados y etiquetados manualmente. La etiqueta de clase está representada como un número (`1, 2, 3, 4, 5`), el cual corresponde a las señas descritas anteriormente.

## 2. Estructura General del Repositorio

La organización del repositorio se realiza de la siguiente forma:

```
.
├── Metadata
│   ├── data.csv
│   ├── data_curated.csv
│   ...
├── Modeling
│   ├── Preprocessing
│   │   ├── holistic.py
│   │   ├── transform.py
│   │   └── utils.py
│   ├── utils
│   │   ├── load.py
│   │   ├── metrics.py
│   │   └── tunner.py
│   ├── main.py
│   ├── modeling.py
│   ├── Experiments_CNN_Models.ipynb
│   ├── Experiments_GRU.ipynb
│   ├── Experiments_LSTM.ipynb
│   └── Experiments_RNN.ipynb
├── Raw_Data
│   ├── 1
│   │   ├── S1V1C1M1A Clip1.mp4
│   │   ├── ...
│   ├── 2
│   ├── 3
│   ├── 4
│   └── 5
├── Clean_Data
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   └── 5
├── Data_Augmented_Clean
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   └── 5
└── requirements.txt (opcional/futuro)
```

- **Metadata**: Contiene la información de las rutas y clases en formato `.csv`.
- **Modeling**: Incluye todos los scripts de preprocesamiento, utilidades (carga de datos, métricas, tunning), un archivo principal para la ejecución de pipelines (`main.py`), y notebooks de experimentación con diferentes modelos.
- **Raw_Data**: Estructura de carpetas numeradas (`1, 2, 3, 4, 5`), donde cada carpeta representa una clase y contiene sus respectivos videos `.mp4`.
- **Clean_Data**: Almacén de datos preprocesados, donde cada archivo `.npy` contiene los keypoints de un video.
- **Data_Augmented_Clean**: Almacén de datos aumentados que, en caso de requerirse, contendrá las versiones aumentadas (rotación, traslación, espejado, etc.) de los keypoints.

## 3. Descripción de los Datos

1. **Dataset**:
   - 1500 videos en total, distribuidos en 5 clases:
     - Clase 1: *Café*
     - Clase 2: *Aromática*
     - Clase 3: *Con Gusto*
     - Clase 4: *Qué necesita*
     - Clase 5: *Hola*
   - 300 videos por clase (dataset balanceado).
   - Videos grabados con diferentes personas (M1, M2, M3, M4, M5).

2. **Metadata**:
   - **`data.csv`**: Mapea cada archivo de video `.mp4` a su clase numérica.  
   - **`data_curated.csv`**: Versión “curada” o actualizada de `data.csv`, donde en lugar de referenciar archivos `.mp4` se hace referencia a archivos `.npy` ya procesados.

### Ejemplo de `data.csv`

```csv
File,Class
S1V3C1M3A Clip166.mp4,1
S1V1C1M1A Clip18.mp4,1
...
```

### Ejemplo de `data_curated.csv`

```csv
File,Class
S1V3C1M3A Clip166.npy,1
S1V1C1M1A Clip18.npy,1
...
```

## 4. Pipeline de Procesamiento

El flujo de trabajo para convertir los videos `.mp4` a keypoints `.npy` es como sigue:

1. **Creación de dataframes** (`data.csv` y `data_curated.csv`):
   - Se enumeran todas las carpetas de `Raw_Data` para generar `data.csv`.
   - Se actualizan las extensiones de video a `.npy` en `data_curated.csv`.

2. **Transformación de los videos** (uso de Mediapipe Holistic en `holistic.py`):
   - Se cargan los videos con **OpenCV** (`cv2.VideoCapture`).
   - Se utiliza la solución **Holistic** de **MediaPipe** para extraer landmarks de **manos**, **pose** y **rostro**.
   - Se convierten los frames de BGR a RGB para ser procesados por Mediapipe.
   - Se almacenan los keypoints en arrays de NumPy, con la siguiente estructura:
     - Pose: 33 landmarks × 4 valores (x, y, z, visibility) = 132 valores.
     - Cara (Face): 468 landmarks × 3 valores (x, y, z) = 1404 valores.
     - Mano izquierda (LH): 21 landmarks × 3 valores (x, y, z) = 63 valores.
     - Mano derecha (RH): 21 landmarks × 3 valores (x, y, z) = 63 valores.
     - **Total**: 132 + 1404 + 63 + 63 = **1662** valores por frame.

3. **Almacenamiento en formato `.npy`**:
   - Cada video se convierte en un tensor (array 2D) de dimensiones `(n_frames, 1662)`.
   - Se guarda el resultado como archivo NumPy (`.npy`) dentro de la carpeta correspondiente a su clase en `Clean_Data`.

4. **Transformación adicional / Data Augmentation** (opcional):
   - Se pueden aplicar transformaciones espaciales o temporales para aumentar la diversidad de datos.
   - Los archivos resultantes se almacenan en `Data_Augmented_Clean`.

### `holistic.py`

```python
import cv2
import mediapipe as mp
import numpy as np

_mp_drawing = mp.solutions.drawing_utils
_mp_holistic = mp.solutions.holistic

def transform_video(video: cv2.VideoCapture):
    """
    Return keypoints from the video.
    Input: video : cv2.VideoCapture
    Output: key_points : np.array
    """
    with _mp_holistic.Holistic(static_image_mode=False,
                               model_complexity=2) as holistic:
        return _process_video(video, holistic)

def _process_video(video, holistic):
    processed_points = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        processed_points.append(_extract_keypoints(results=results))
    return np.array(processed_points)

def _extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() \
                     if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() \
                     if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() \
                   if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() \
                   if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])
```

> **Nota**: Se incluyen landmarks de la cara (face) porque, en la experiencia con Lengua de Señas, las expresiones faciales y movimientos del rostro juegan un papel relevante para la correcta interpretación.

## 5. Scripts de Preprocesamiento

### `transform.py`

```python
import pandas as pd
import cv2
import numpy as np
from Preprocessing.holistic import transform_video
from progress.bar import Bar

df = pd.read_csv("../Metadata/data.csv")
df.set_index("File", inplace=True)

def process_data():
    with Bar('Processing...', max=df.size) as bar:
        for file_name, class_name in df.iterrows():
            # Construimos la ruta del video
            video_path = f"../Raw_Data/{class_name.iloc[0]}/{file_name}"
            result = transform_video(cv2.VideoCapture(video_path))
            # Guardamos el resultado en Clean_Data
            np.save(f"../Clean_Data/{class_name.iloc[0]}/{file_name[:-4]}.npy", result)
            bar.next()
```

### `utils.py` (Preprocessing)

```python
import os
import pandas as pd

def create_dataframe():
    classes = ['1','2','3','4','5']
    data = []
    for c in classes:
        for f in os.listdir(f'../Raw_Data/{c}'):
            data.append({ 
                "File": f,
                "Class": c
            })
    data = pd.DataFrame(data)
    data.to_csv('../Metadata/data.csv', index=False)

def mp4_to_npy_dataframe():
    data = pd.read_csv("../Metadata/data.csv")
    len_data = len(data)
    for file_row in range(len_data):
        data.at[file_row, "File"] = data.at[file_row,"File"][:-4]+".npy"
    data.to_csv('../Metadata/data_curated.csv', index=False)
```

## 6. División de Datos y Carga

Se siguen dos metodologías principales para la división de datos, enfocadas en **evitar el sobreajuste** y **evaluar la capacidad de generalización**:

1. **Split basado en etiquetas `M1`, `M2`, `M3`, `M4`, `M5`**:
   - Cada letra/etiqueta (M1, M2, etc.) se asocia a los videos grabados por una **persona** diferente.
   - Se puede apartar, por ejemplo, todos los videos que contengan la etiqueta `M1` en su nombre de archivo como set de validación, o mezclarlos en un split aleatorio.

2. **Random cutting**:
   - Se eligen subsecuencias (frames) aleatorias de longitud `size` para cada video, de modo que se normalicen las dimensiones de entrada al modelo.

### `load.py`

```python
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
from typing import List
import random

def load_data_methodology_cutting(df, size, val_split_label):
    """
    Carga datos dividiendo en entrenamiento y validación en función de val_split_label.
    ...
    """
    # Ejemplo de lógica (ver script completo arriba)

def load_data_methodology_random_cutting(df: pd.DataFrame, val_split_labels: List[str], size: int, seed: int = 42):
    """
    Lógica similar, pero realizando splits aleatorios y recortes.
    ...
    ```
```

## 7. Métricas de Evaluación

Se emplean las siguientes métricas con Keras:

- **Precision**  
- **Recall**  
- **F1 Score** (Se calcula en batch; según la configuración, podría aproximar un F1 macro).

Además, se ha implementado una función para graficar la **matriz de confusión**. Esto es importante en problemas multiclase, pues permite identificar qué clases se confunden con mayor frecuencia.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow.keras.backend as K

def precision(y_true, y_pred):
    # ...
def recall(y_true, y_pred):
    # ...
def f1(y_true, y_pred):
    # ...
def confusion_matrix_plot(y_true, y_pred):
    # ...
```

> **Nota**: Dada la naturaleza multiclase (5 clases), se recomienda en la documentación detallar el tipo de promedio usado (macro, micro, weighted) para el F1 final.

## 8. Modelos y Notebooks de Experimentos

Dentro de la carpeta `Modeling`, existen varios cuadernos Jupyter que muestran experimentos con distintas arquitecturas: **CNN, RNN, LSTM, GRU**. El flujo típico en un notebook es:

1. Carga de datos (`load_data_methodology_random_cutting` o variantes).
2. Definición del modelo secuencial (capas de LSTM, GRU, Dense, etc.).
3. Compilación del modelo con el optimizador deseado (en la mayoría de experimentos se ha utilizado **Nadam**).
4. Ajuste de hiperparámetros con la clase `CVTuner`, la cual implementa un **Bayesian Optimization** y un esquema de **Cross-Validation** (`MultilabelStratifiedKFold`).

### Ejemplo: `Experiments_GRU.ipynb`

```python
import pandas as pd
import numpy as np
from utils.load import load_data_methodology_random_cutting
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils.metrics import f1
from utils.tuner import CVTuner
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import joblib
import keras_tuner
import tensorflow as tf

df = pd.read_csv("../Metadata/data_curated.csv")
X_train, y_train, X_val, y_val = load_data_methodology_random_cutting(df, ["M1"], 30)

cv = MultilabelStratifiedKFold(n_splits=5)
joblib.dump(cv, './cv/cv.joblib')

def build_model_LSTM(hp):
    model = Sequential()
    # ...
    model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=[f1])
    return model

tuner_LSTM = CVTuner(
    data_cv=cv,
    goal='f1',
    hypermodel=build_model_LSTM,
    oracle=keras_tuner.oracles.BayesianOptimizationOracle(
        objective=keras_tuner.Objective('f1', direction="max"),
        max_trials=5
    ),
    directory='./experiments/',
    project_name='LSTM',
    overwrite=True
)

tuner_LSTM.search(X_train, y_train, 64, epochs=50)
df_LSTM = pd.DataFrame(tuner_LSTM.trial_scores)
```

## 9. Ejecución Principal

Para generar los `.csv`, curarlos y procesar los videos a `.npy`, se ejecuta:

```bash
cd Modeling
python main.py
```

El contenido de `main.py`:

```python
from Preprocessing.transform import process_data
from Preprocessing.utils import create_dataframe, mp4_to_npy_dataframe

print("Creating dataframes...")
create_dataframe()
mp4_to_npy_dataframe()
print("Done creating dataframes.")
print("Start to process data...")
process_data()
```

## 10. Infraestructura de Cómputo y Reproducibilidad

- El entrenamiento se realiza **sobre CPU**; por lo tanto, los tiempos de entrenamiento pueden ser elevados para redes profundas (p.ej., LSTM con muchos parámetros).
- Se recomienda tener un entorno que contenga:
  - **Python 3.7+**
  - **OpenCV (cv2)**
  - **mediapipe**
  - **TensorFlow / Keras**
  - **NumPy, Pandas, scikit-learn**, etc.

> Un archivo `requirements.txt` o un entorno Conda facilitaría la reproducibilidad de los experimentos. Se sugiere añadirlo en el futuro con la versión específica de las librerías.

## 11. Posibles Extensiones Futuras

1. **Data Augmentation**: 
   - Actualmente se contempla la carpeta `Data_Augmented_Clean`, aunque no se ha detallado un script específico de augmentación. Se podrían implementar transformaciones de las series de keypoints (p.ej., jitter en la dimensión temporal, espejado simétrico de manos, etc.).
2. **Más Clases**: 
   - A pesar de que no se planea en este momento, se podría ampliar a otras señas de la LSC.
3. **Mayor Variedad de Optimizadores**:
   - Se ha utilizado **Nadam** principalmente. Se podrían evaluar **Adam**, **RMSProp**, **SGD** con diferentes tasas de aprendizaje, comparar resultados y documentarlos.
4. **Visualización de Landmarks**:
   - Como trabajo extra, se podría generar un script para visualizar en tiempo real los landmarks de MediaPipe en cada video, validando que la captura sea correcta.

## 12. Conclusiones

- El proyecto construye un pipeline **integral** para la clasificación de señas de la LSC, desde la adquisición de videos (`Raw_Data`) hasta la extracción de keypoints con MediaPipe (`Clean_Data`) y, posteriormente, la evaluación de diferentes arquitecturas de redes neuronales.
- La inclusión de **landmarks faciales** es importante dada la naturaleza expresiva de la Lengua de Señas.
- La documentación y organización por carpetas permiten expandir la solución a futuras clases o nuevas personas sin tener que modificar la base del pipeline.
- La precisión, recall y F1 se utilizan como métricas estándar para medir el desempeño en un escenario multiclase.

---

**© 2024 - SignAI UdeA.**  
Para cualquier duda o contribución, revisar los notebooks de experimentación o contactar a los autores del repositorio.  
