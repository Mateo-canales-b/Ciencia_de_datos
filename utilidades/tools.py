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
from scipy.stats import kurtosis, skew, ttest_ind, t, shapiro
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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

def make_viridis_palette(k=12, span=(0.15, 0.95)):
    cmap = plt.colormaps.get_cmap('viridis')   # ✅ forma nueva y segura
    return cmap(np.linspace(span[0], span[1], k))

VIRIDIS = make_viridis_palette(k=12)

def boxplot_viridis(
    df, col, n=0, *,
    whis=(5, 95),          # bigotes por percentiles (más robusto que 1.5*IQR si hay colas largas)
    showfliers=False,      # oculta outliers por defecto (puedes poner True)
    log=False,             # escala log opcional
    title_prefix="Distribución de"
):
    """
    Boxplot horizontal con Viridis, configurable y limpio.
    - df: DataFrame
    - col: nombre de la columna numérica a graficar
    - n: índice para elegir color en la paleta VIRIDIS
    - whis: "1.5" para usar el 1.5*IQR, o "(5,95)" para más robustez 
    """
    serie = df[col].dropna().values
    c = VIRIDIS[n % len(VIRIDIS)]  # color viridis ciclado
    label = col.replace("_", " ")

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        serie,
        vert=False,
        patch_artist=True,
        whis=whis,
        #showfliers=showfliers,
        boxprops=dict(facecolor=c, edgecolor=c, linewidth=1.6, alpha=0.95),
        medianprops=dict(color="red", linewidth=2.2),
        whiskerprops=dict(color=c, linewidth=1.6),
        capprops=dict(color=c, linewidth=1.6),
        flierprops=dict(marker="o", markerfacecolor="orange", markeredgecolor="black",
                        markersize=6, alpha=0.6, linestyle="none")
    )

    if log:
        plt.xscale("log")
        plt.xlabel(f"{label} (escala log)")
    else:
        plt.xlabel(label)

    plt.title(f"{title_prefix} {label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def detectar_outliers_iqr(
    df: pd.DataFrame,
    columna: str,
    n,
    graficar: bool = True,
    inplace: bool = False
):
    """
    Detecta y filtra outliers (IQR). Si inplace=True, modifica df en sitio.
    
    Returns:
        outliers (pd.DataFrame): Filas detectadas como outliers.
        df_filtrado (pd.DataFrame): DataFrame sin outliers (o None si inplace=True).
        limites (dict): Q1, Q3, IQR, límites.
    """
        # 1) (Opcional) Gráfico
    if graficar:
        boxplot_viridis(df, columna, n=n, whis=1.5)

    # 2) Cuartiles e IQR
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1

    # 3) Límites de Tukey
    li = Q1 - 1.5 * IQR
    ls = Q3 + 1.5 * IQR

    # 4) Máscara de outliers (NaN no cuenta como outlier)
    mask_out = (df[columna] < li) | (df[columna] > ls)

    # 5) Subsets
    outliers = df.loc[mask_out].copy()
    df_filtrado = df.loc[~mask_out].copy()

    # 6) Métricas y salida formateada
    total_validos = df[columna].notna().sum()
    num_out = int(mask_out.sum())
    pct_out = (num_out / total_validos * 100.0) if total_validos > 0 else 0.0

    limites = {"Q1": Q1, "Q3": Q3, "IQR": IQR,
               "limite_inferior": li, "limite_superior": ls}    
    
    # 7) Impresión de resultados
    print(f"Outliers detectados en {columna}: {num_out} ({pct_out:.2f}%)")
    print("Límites:")
    for k, v in limites.items():
        print(f"  {k}: {v:.2f}")
    print("Tamaño sin outliers:", df_filtrado.shape)

    # 8) In-place o retorno
    if inplace:
        # Modifica el df original en el caller usando índices (sin reasignar la variable local)
        df.drop(df.index[mask_out], inplace=True)
        return outliers, None, limites
    else:
        return outliers, df_filtrado, limites


# ========= Utilidad: Shapiro con manejo de errores =========
def shapiro_safe(x):
    """
    Ejecuta Shapiro-Wilk ignorando NaN y maneja casos con pocos datos o constantes.
    Retorna (stat, pvalue, msg).
    """
    x = pd.Series(x).dropna().values
    if len(x) < 3:
        return np.nan, np.nan, "muestra insuficiente (<3)"
    # Shapiro requiere que no sean todos iguales
    if np.allclose(x, x[0]):
        return np.nan, np.nan, "valores constantes"
    try:
        stat, pval = shapiro(x)
        return stat, pval, None
    except Exception as e:
        return np.nan, np.nan, f"error: {e}"


# ========= 1) HISTOGRAMA + KDE =========
def plot_hist_kde(df, col, ax=None, *, bins='sturges', n=0, title_prefix="Distribución"):
    """
    Histograma + KDE para la columna 'col'.
    """
    serie = df[col].dropna()
    c = VIRIDIS[n % len(VIRIDIS)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))

    sns.histplot(serie, kde=True, bins=bins, ax=ax, color=c)
    ax.set_title(f"{title_prefix}: {col.replace('_',' ')}", weight="bold", fontsize=12)
    ax.set_xlabel("")

    return ax


# ========= 2) QQ-PLOT (con p-value de Shapiro opcional) =========
def plot_qq(df, col, ax=None, *, n=0, show_pvalue=True, alpha=0.05):
    """
    QQ-plot contra normal. Si show_pvalue=True, calcula Shapiro y lo muestra en el título.
    """
    serie = df[col].dropna()
    c = VIRIDIS[n % len(VIRIDIS)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))

    # statsmodels dibuja sus propios puntos; ajustamos línea 45
    sm.qqplot(serie, ax=ax, fit=True, line="45", markerfacecolor=c, markeredgecolor="black", alpha=0.7)
    ax.set_xlabel("Cuantiles teóricos")
    ax.set_ylabel("Cuantiles muestrales")

    title = col.replace("_"," ")
    if show_pvalue:
        _, pval, err = shapiro_safe(serie)
        if err is None and not np.isnan(pval):
            normal_msg = "no normal" if pval < alpha else "no se descarta normalidad"
            title = f"{title}\nShapiro p={pval:.3f} ({normal_msg})"
        else:
            title = f"{title}\nShapiro: {err}"
    ax.set_title(title, weight="bold", fontsize=12)

    return ax


# ========= 3) BOXPLOT VIRIDIS (horizontal) =========
def boxplot_viridis(
    df, col, ax=None, n=0, *,
    whis=(5, 95),
    showfliers=False,
    log=False,
    title_prefix="Distribución de"
):
    """
    Boxplot horizontal con Viridis.
    """
    serie = df[col].dropna().values
    c = VIRIDIS[n % len(VIRIDIS)]
    label = col.replace("_", " ")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))

    ax.boxplot(
        serie,
        vert=False,
        patch_artist=True,
        whis=whis,
        showfliers=showfliers,
        boxprops=dict(facecolor=c, edgecolor=c, linewidth=1.6, alpha=0.95),
        medianprops=dict(color="red", linewidth=2.2),
        whiskerprops=dict(color=c, linewidth=1.6),
        capprops=dict(color=c, linewidth=1.6),
        flierprops=dict(marker="o", markerfacecolor="orange", markeredgecolor="black",
                        markersize=6, alpha=0.6, linestyle="none")
    )

    if log:
        ax.set_xscale("log")
        ax.set_xlabel(f"{label} (escala log)")
    else:
        ax.set_xlabel(label)

    ax.set_yticks([])
    ax.set_title(f"{title_prefix} {label}", fontsize=12, weight="bold")
    return ax


# ========= 4) FUNCIÓN MAESTRA: los 3 gráficos en una sola imagen =========
def triple_plot(df, col, *, n=0, bins='sturges', alpha=0.05, whis=(5,95), showfliers=False, log=False):
    """
    Dibuja en una figura: Histograma+KDE | QQ-plot | Boxplot viridis.
    Reutiliza las funciones anteriores.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_hist_kde(df, col, ax=axs[0], bins=bins, n=n)
    plot_qq(df, col, ax=axs[1], n=n, show_pvalue=True, alpha=alpha)
    boxplot_viridis(df, col, ax=axs[2], n=n, whis=whis, showfliers=showfliers, log=log)

    fig.suptitle(f"Distribución de '{col}'", fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()



def elbow_auto_plot(X, k_min=2, k_cap=10, scale=True, scaler="standard", 
                    method="absolute", palette=None, seed=42, show=True):
    """
    Método del codo automatizado para KMeans.
    
    Parámetros
    ----------
    X : array o DataFrame
        Datos de entrada.
    k_min : int
        Número mínimo de clusters a probar (default=2).
    k_cap : int
        Máximo k a evaluar (se ajusta a n_samples-1 si es más grande).
    scale : bool
        Si True, escala los datos antes de aplicar KMeans.
    scaler : str
        "standard" (StandardScaler) o "minmax" (MinMaxScaler).
    method : str
        Método para elegir el codo: "absolute" o "relative".
    palette : lista o None
        Paleta de colores (ej. VIRIDIS).
    seed : int
        Semilla aleatoria para reproducibilidad.
    show : bool
        Si True, muestra el gráfico.
    """
    X_arr = np.asarray(X)
    n_samples = X_arr.shape[0]

    # Escalado opcional
    if scale:
        if scaler == "standard":
            X_arr = StandardScaler().fit_transform(X_arr)
        elif scaler == "minmax":
            X_arr = MinMaxScaler().fit_transform(X_arr)

    # Rango de k
    k_max = max(k_min, min(k_cap, n_samples - 1))
    k_values = list(range(k_min, k_max + 1))

    # Inercias
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        km.fit(X_arr)
        inertias.append(km.inertia_)

    # Método del codo
    if len(inertias) >= 2:
        diffs = np.diff(inertias) * -1           # mejoras positivas
        rel_diffs = diffs / np.array(inertias[:-1])
        # Elegir el codo según método
        if method == "relative":
            elbow_idx = int(np.argmax(rel_diffs))
        else:  # "absolute" por defecto
            elbow_idx = int(np.argmax(diffs))
        elbow_k = k_values[elbow_idx + 1]
    else:
        diffs = np.array([])
        rel_diffs = np.array([])
        elbow_k = k_values[0]


    # Crear DataFrame
    df_out = pd.DataFrame({
        "k": k_values,
        "inertia": inertias,
        "delta_inertia": [np.nan] + diffs.tolist(),
        "delta_relative": [np.nan] + rel_diffs.tolist(),
        "method": method,
        "scaled": scale,
        "scaler": scaler
    }).set_index("k")

    df_out["is_elbow"] = df_out.index == elbow_k

    # Paleta 
    line_color = "black"
    line_colordos= "black"
    if palette is not None:
        try:
            line_color = palette[5]   # toma un color intermedio si existe
            line_colordos= palette[0]
        except Exception:
            line_color = "black"
            line_colordos= "black"
    # Gráfico
    if show:
        vir = plt.cm.viridis
        colors = vir(np.linspace(0, 1, len(k_values)))
        colors2 = vir(np.linspace(0, 1, len(k_values)-1)) if len(k_values) > 1 else None

        fig, ax1 = plt.subplots(figsize=(8,6))
        # Inercia
        ax1.plot(k_values, inertias, marker="o", color=line_color, label="Inercia total")
        ax1.set_xlabel("Número de clusters (k)")
        ax1.set_ylabel("Inercia (SSE)", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.axvline(elbow_k, linestyle="--", alpha=0.6, color=line_colordos,
                    label=f"Codo sugerido: k={elbow_k}")

        # Deltas
        ax2 = ax1.twinx()
        if diffs.size:
            ax2.bar(k_values[1:], diffs, color=colors2, alpha=0.25, label="Δ Inercia")
        ax2.set_ylabel("Δ Inercia", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        # Leyenda
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="best")

        plt.title(f"Método del codo ({method}, scale={scale}, scaler={scaler})", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.show()

    return df_out, elbow_k
def silhouette_auto_plot(X, k_min=2, k_cap=10, scale=True, scaler="standard",
                         seed=42, palette=None, show=True):
    """
    Evalúa Silhouette vs k para KMeans y devuelve (df_resultados, best_k).

    Parámetros
    ------
    X : array-like o DataFrame (solo numérico)
    k_min : int, mínimo k a evaluar (>=2)
    k_cap : int, máximo k a intentar (cap superior; se ajusta a n_samples-1)
    scale : bool, si True escala X antes (recomendado para KMeans)
    scaler : {"standard","minmax"}
    seed : int, semilla de reproducibilidad
    palette : lista o None, Paleta de colores (ej. VIRIDIS).
    show : bool, si True grafica Silhouette vs k

    Retorna
    ------
    df_out : DataFrame indexado por k con columnas:
             ["silhouette", "n_clusters", "scaled", "scaler", "is_best"]
    best_k : int, k con mayor coeficiente de silueta
    """
    # saneamiento básico
    X_arr = np.asarray(X, dtype=float)
    # quita filas con NaN/inf si existieran
    mask = np.isfinite(X_arr).all(axis=1)
    X_arr = X_arr[mask]
    n_samples = X_arr.shape[0]

    if n_samples < 3:
        raise ValueError("Se requieren ≥3 muestras para calcular silueta.")

    # escalado opcional
    if scale:
        if scaler == "standard":
            X_arr = StandardScaler().fit_transform(X_arr)
        elif scaler == "minmax":
            X_arr = MinMaxScaler().fit_transform(X_arr)

    # rango de k válido
    k_max = max(k_min, min(k_cap, n_samples - 1))
    if k_max < 2:
        raise ValueError("Con tan pocas muestras no es posible evaluar k≥2.")
    k_values = list(range(k_min, k_max + 1))

    # cálculo de silueta
    sil_scores = []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        labels = km.fit_predict(X_arr)
        # silueta válida solo si hay >1 cluster y ningún cluster vacío
        if len(np.unique(labels)) > 1:
            s = silhouette_score(X_arr, labels)
        else:
            s = np.nan
        sil_scores.append(s)

    # mejor k (ignorando NaN)
    if np.all(np.isnan(sil_scores)):
        best_k = k_values[0]
    else:
        best_idx = int(np.nanargmax(sil_scores))
        best_k = k_values[best_idx]

    # DataFrame de salida
    df_out = pd.DataFrame({
        "k": k_values,
        "silhouette": sil_scores,
        "n_clusters": k_values,
        "scaled": scale,
        "scaler": scaler
    }).set_index("k")
    df_out["is_best"] = df_out.index == best_k

    # Paleta 
    line_color = "black"
    if palette is not None:
        try:
            line_color = palette[6]   
        except Exception:
            line_color = "black"
    # gráfico opcional
    if show:
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
        plt.figure(figsize=(8,5))
        plt.plot(k_values, sil_scores, marker="o", linewidth=1.5, color=line_color)
        
        # marca best_k
        best_s = df_out.loc[best_k, "silhouette"]
        if np.isfinite(best_s):
            plt.scatter(best_k, best_s, s=220, marker="X",
                        edgecolors="k", linewidths=2, color="red",
                        label=f"Mejor k: {best_k} (s={best_s:.3f})")
        plt.title(f"Silueta vs k (scale={scale}, scaler={scaler})",
                  fontsize=14, weight="bold")
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Silueta (−1 a 1)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_out, best_k