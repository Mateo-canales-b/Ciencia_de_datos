# tools_extra.py
# Librerías opcionales para ciencia de datos

# Visualización interactiva
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px, go = None, None

# Gradient Boosting
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cat
except ImportError:
    cat = None

# Desbalance de clases
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

# Motores de datos / IO
try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pyarrow as pa
except ImportError:
    pa = None

# Optimización de hiperparámetros
try:
    import optuna
except ImportError:
    optuna = None

# Explicabilidad de modelos
try:
    import shap
except ImportError:
    shap = None

# Series de tiempo
try:
    import pmdarima as pm
except ImportError:
    pm = None
