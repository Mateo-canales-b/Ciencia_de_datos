# tools.py 

# Manejo de datos
import pandas as pd
import numpy as np
from tabulate import tabulate

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Estadística y análisis de series temporales
from scipy import stats
from scipy.stats import kurtosis, skew
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# Machine Learning con scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Deep Learning con TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Kaggle utilities
import kagglehub

# Calidad de vida en notebooks
import warnings

# Ajustes por defecto razonables para EDA
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 120)

sns.set_theme()
warnings.filterwarnings("ignore")

# Spark (PySpark)
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F, types as T, Window as W
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    OneHotEncoder as SparkOneHotEncoder,
    StandardScaler as SparkStandardScaler,
    MinMaxScaler as SparkMinMaxScaler,
    Imputer as SparkImputer,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.classification import (
    LogisticRegression as SparkLogisticRegression,
    RandomForestClassifier as SparkRandomForestClassifier,
    GBTClassifier as SparkGBTClassifier,
)
from pyspark.ml.regression import (
    RandomForestRegressor as SparkRandomForestRegressor,
    GBTRegressor as SparkGBTRegressor,
)

def get_spark(app_name: str = "CienciaDeDatos", local_cores: str = "*") -> SparkSession:
    """Crear o recuperar una SparkSession local con nivel de log reducido.

    Parameters
    ----------
    app_name : nombre de la aplicación a mostrar en Spark UI
    local_cores : número de cores locales ("*" usa todos)
    """
    spark = (
        SparkSession.builder
        .master(f"local[{local_cores}]")
        .appName(app_name)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
def tabla(dfx, fmt='fancy_grid', decimals=2, title=None):
    """
    Muestra un DataFrame o Series en formato tabulado con título opcional.
    
    Parámetros:
    dfx : DataFrame o Series
    fmt : str, formato de tabulate (default: 'fancy_grid')
    decimals : int, número de decimales (default: 2)
    title : str, título de la tabla (default: None)
    """
    # Si es Series → convertir a DataFrame
    if isinstance(dfx, pd.Series):
        dfx = dfx.to_frame(name="Valor")

    # Redondear si son datos numéricos
    try:
        dfx = dfx.round(decimals)
    except:
        pass  # ignora si no aplica

    # Imprimir título si existe
    if title:
        print(f"\n {title}\n" + "-" * (len(title) + 2))

    print(tabulate(dfx, headers="keys", tablefmt=fmt))
