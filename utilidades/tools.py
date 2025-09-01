# tools.py 

# Manejo de datos
import pandas as pd
import numpy as np

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
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
