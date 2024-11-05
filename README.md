
# Proyecto de NLP

Este proyecto se enfoca en el **Procesamiento de Lenguaje Natural (NLP)** utilizando técnicas avanzadas de **machine learning** y **deep learning** para tareas como análisis de sentimiento, clasificación de texto, y generación de lenguaje. Usamos herramientas como `nltk`, `spacy` y `transformers`.

## Estructura del Proyecto

```plaintext
├── data/
│   ├── raw/               # Datos crudos
│   ├── processed/         # Datos procesados listos para modelado
├── notebooks/
│   ├── exploracion.ipynb  # Exploración y visualización de datos
│   ├── model_training.ipynb # Entrenamiento y evaluación de modelos
├── src/
│   ├── preprocessing.py   # Código de preprocesamiento de texto
│   ├── models.py          # Definición de modelos de NLP
│   └── utils.py           # Funciones auxiliares
├── main.py                # Script principal del proyecto
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Descripción del proyecto
```

## Objetivos del Proyecto

- **Preprocesamiento de texto:** Limpieza, tokenización, lematización y eliminación de stopwords.
- **Modelos de clasificación:** Entrenar modelos supervisados como Naive Bayes, SVM y transformers para la clasificación de texto.
- **Generación de texto:** Implementar modelos avanzados como GPT para la creación automática de texto coherente.
- **Evaluación de modelos:** Medir el rendimiento con métricas como accuracy, precision, recall y F1 score.

## Instalación

Para ejecutar este proyecto, clona el repositorio y luego instala las dependencias usando `pip`:

```bash
git clone https://github.com/tu-usuario/NLP_Projecto.git
cd NLP_Projecto
pip install -r requirements.txt
```

### Dependencias clave:
- **nltk:** Procesamiento de lenguaje natural.
- **spacy:** Tokenización avanzada y lematización.
- **transformers:** Modelos avanzados como BERT y GPT.
- **scikit-learn:** Algoritmos clásicos de machine learning.
- **pandas y numpy:** Manipulación de datos y operaciones numéricas.

## ⚙️ Uso del Proyecto

### 1. Preprocesamiento de Datos

Limpia y transforma los datos utilizando el script `preprocessing.py`:

```bash
python src/preprocessing.py --input data/raw/dataset.csv --output data/processed/cleaned_data.csv
```

### 2. Entrenamiento de Modelos

Entrena un modelo de clasificación o generación de texto usando los notebooks o ejecutando el script principal:

```bash
python main.py --train --model svm --input data/processed/cleaned_data.csv
```

### 3. Evaluación del Modelo

Evalúa el rendimiento del modelo con este comando:

```bash
python main.py --evaluate --model svm --input data/processed/cleaned_data.csv
```

### 4. Generación de Texto

Genera texto usando un modelo como GPT:

```bash
python main.py --generate --model gpt --input "Completa esta frase:"
```

## 📊 Notebooks

Los notebooks incluidos en el proyecto, que puedes ejecutar localmente o en Google Colab, son:

- `exploracion.ipynb`: Análisis exploratorio de los datos.
- `model_training.ipynb`: Entrenamiento y evaluación de modelos NLP.

## 🔧 Configuración de Entrenamiento

Al ejecutar el script `main.py`, puedes especificar el tipo de modelo y el dataset a utilizar. Por ejemplo, para entrenar un modelo SVM:

```bash
python main.py --train --model svm --input data/processed/cleaned_data.csv
```

O para usar un modelo basado en transformers como BERT:

```bash
python main.py --train --model bert --input data/processed/cleaned_data.csv
```

## 🔍 Evaluación de Resultados

Las métricas de evaluación y gráficos de rendimiento se almacenan automáticamente en la carpeta `results/`. Incluyen:

- Matrices de confusión.
- Reportes de clasificación.
- Comparación de rendimiento entre modelos.

## Contribuciones

Si deseas contribuir a este proyecto:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz un commit (`git commit -m 'Agregar nueva funcionalidad'`).
4. Sube tus cambios (`git push origin feature/nueva-funcionalidad`).
5. Abre un pull request.

## Futuras Mejoras

- Implementar **transfer learning** utilizando modelos preentrenados como BERT y GPT.
- Añadir funciones de **resúmenes automáticos** y técnicas avanzadas de **traducción automática**.
- Optimización y ajuste de hiperparámetros usando **GridSearch** o **Bayesian Optimization**.
