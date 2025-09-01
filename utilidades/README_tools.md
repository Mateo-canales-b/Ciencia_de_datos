# README — Entorno de Ciencia de Datos con `tools.py`

## 📦 ¿Qué es `tools.py`?
`tools.py` centraliza los **imports** más usados en ciencia de datos para que, en tus notebooks, baste con:

```python
# Si el notebook está en subcarpetas, ajusta el path a la raíz del proyecto
import sys, os
sys.path.insert(0, os.path.abspath("../.."))

from tools import *   # carga pandas, numpy, sklearn, statsmodels, etc.
```

Con una sola línea, tu entorno queda listo (estilo de gráficos, opciones de pandas y librerías clave).

---

## 🗂️ Librerías incluidas (y para qué se usan)

### Manejo de datos
- **pandas (`pd`)**: tablas, joins, lectura/escritura de CSV/Excel/Parquet.
- **numpy (`np`)**: arreglos, operaciones numéricas vectorizadas.

### Visualización
- **matplotlib (`plt`)**: gráficos base, control fino de figuras.
- **seaborn (`sns`)**: visualizaciones estadísticas de alto nivel; setea tema por defecto.

### Estadística y series de tiempo
- **scipy (`stats`, `kurtosis`, `skew`)**: tests estadísticos, distribuciones.
- **statsmodels (`sm`, `smf`)**: modelos estadísticos con y sin fórmulas (OLS, GLM, etc.).
- **statsmodels graphics/tsa (`plot_acf`, `acf`)**: ACF/diagnósticos en series.

### Machine Learning (scikit-learn)
- **Modelos**: `LinearRegression`
- **Métricas**: `mean_squared_error`, `mean_absolute_error`, `r2_score`, `roc_auc_score`, `roc_curve`, `precision_recall_fscore_support`, `accuracy_score`, `confusion_matrix`, `classification_report`
- **Validación**: `train_test_split`, `GridSearchCV`, `TimeSeriesSplit`, `StratifiedKFold`
- **Preprocesamiento**: `MinMaxScaler`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`
- **Pipelines**: `Pipeline`, `ColumnTransformer`

### Deep Learning
- **tensorflow / keras (`tf`, `keras`, `layers`)**: redes neuronales, entrenamiento en GPU/CPU.

### Utilidades diversas
- **kagglehub**: descarga datasets de Kaggle (autenticación con token).
- **warnings**: silenciar/gestionar advertencias para notebooks.

---

## ⚙️ Configuración por defecto en notebooks
- `pandas` muestra más columnas/ancho cómodo para EDA.
- `seaborn.set_theme()` aplica estilo limpio.
- `warnings.filterwarnings("ignore")` oculta advertencias comunes.

---

## 🚀 Ejemplo mínimo de uso

```python
from tools import *

# Datos demo
df = pd.DataFrame(np.random.randn(200, 3), columns=list("ABC"))

# Visualización
sns.pairplot(df)

# Regresión simple
X = df[["A", "B"]]
y = df["C"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, pred, squared=False))
```

---

## 🧭 Buenas prácticas
- Para **exploración**: `from tools import *` (comodidad y rapidez).
- Para **scripts de producción**: importa **solo** lo necesario (arranque más rápido y dependencias explícitas).
- Si mueves notebooks entre carpetas, añade al inicio:
  ```python
  import sys, os
  sys.path.insert(0, os.path.abspath("../.."))
  ```

---

## 🐢 ¿Impacto en rendimiento?
- **Arranque**: puede tardar un poco más porque carga varias librerías aun si no las usas.
- **Ejecución**: no cambia la velocidad de tus modelos/procesos.
- **Memoria**: algunos paquetes (p. ej. TensorFlow) consumen RAM al cargar.

Si buscas un arranque más ágil en notebooks ligeros, crea un `tools_light.py` con solo `numpy/pandas/matplotlib/seaborn/sklearn base`.

---

## 🔧 Instalación recomendada

```bash
# Base recomendada
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn tensorflow kagglehub
```

> Nota: TensorFlow puede requerir versiones específicas según tu SO/GPU. Si da problemas, prueba una versión concreta (ej. `tensorflow==2.15.*`) o usa `pip install tensorflow-macos` en Mac ARM.

---

# Librerías opcionales que removimos (cuándo usarlas y cómo instalarlas)

### Visualización interactiva
- **plotly (`px`, `go`)**  
  **Úsala para** dashboards o gráficos interactivos (hover, zoom, export HTML).  
  **Instala**: `pip install plotly`

### Gradient Boosting de alto rendimiento
- **xgboost (`xgb`)**  
  **Úsala para** tabular ML con gran performance/velocidad y manejo de missing.  
  **Instala**: `pip install xgboost`
- **lightgbm (`lgb`)**  
  **Úsala para** datasets grandes y features categóricas (muy rápido).  
  **Instala**: `pip install lightgbm`
- **catboost (`cat`)**  
  **Úsala para** categóricas sin one-hot, buen rendimiento “out-of-the-box”.  
  **Instala**: `pip install catboost`

### Desbalance de clases
- **imbalanced-learn (SMOTE)**  
  **Úsala para** sobremuestreo (SMOTE) o estrategias de balanceo en clasificación desbalanceada.  
  **Instala**: `pip install imbalanced-learn`

### Motores de datos / IO
- **polars (`pl`)**  
  **Úsala para** procesamiento tipo pandas pero mucho más rápido (motor en Rust).  
  **Instala**: `pip install polars`
- **pyarrow (`pa`)**  
  **Úsala para** formatos columnares (Arrow/Parquet), interoperabilidad y rapidez I/O.  
  **Instala**: `pip install pyarrow`

### Optimización de hiperparámetros
- **optuna**  
  **Úsala para** búsquedas eficientes de hiperparámetros (TPE, pruning, dashboards).  
  **Instala**: `pip install optuna`

### Explicabilidad de modelos
- **shap**  
  **Úsala para** interpretabilidad (`SHAP values`) en modelos complejos (tree/NN).  
  **Instala**: `pip install shap`

### Series de tiempo (ARIMA auto)
- **pmdarima (`pm`)**  
  **Úsala para** `auto_arima` (elección automática de (p,d,q) y estacionalidad).  
  **Instala**: `pip install pmdarima`
