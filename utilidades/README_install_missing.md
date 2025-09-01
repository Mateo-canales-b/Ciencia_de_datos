# 📦 install_missing.py — Instalador de librerías de ciencia de datos

Este script permite **verificar** e **instalar automáticamente** las librerías más comunes de ciencia de datos en Python.  
Es útil cuando trabajas en distintos equipos o entornos y no quieres preocuparte por recordar todos los paquetes.

---

## 🛠️ Uso básico

En tu terminal, dentro de la carpeta donde esté el archivo:

```bash
python install_missing.py
```

Esto:
- Revisa si tienes instaladas las librerías **base** (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn`, `tensorflow`, `kagglehub`).
- Instala automáticamente las que falten.

---

## ⚙️ Opciones disponibles

```bash
# Ver qué instalaría, sin instalar nada
python install_missing.py --dry-run

# Instalar librerías base (default)
python install_missing.py

# Instalar librerías base + extras (plotly, xgboost, lightgbm, catboost, etc.)
python install_missing.py --extras

# Instalar SOLO extras
python install_missing.py --only-extras

# Forzar actualización a últimas versiones
python install_missing.py --upgrade

# Instalar en modo silencioso (menos logs)
python install_missing.py --quiet
```

---

## 📂 Librerías incluidas

### Base (instaladas siempre)
- `pandas` → manejo de datos tabulares  
- `numpy` → operaciones numéricas  
- `matplotlib`, `seaborn` → visualización  
- `scipy`, `statsmodels` → estadística y modelos  
- `scikit-learn` → machine learning clásico  
- `tensorflow` (o `tensorflow-macos` en Apple Silicon) → deep learning  
- `kagglehub` → descarga datasets de Kaggle  

### Extras (opcionales con `--extras`)
- `plotly` → gráficos interactivos  
- `xgboost`, `lightgbm`, `catboost` → gradient boosting avanzado  
- `imbalanced-learn` → balanceo de clases (SMOTE, etc.)  
- `polars`, `pyarrow` → motores rápidos para big data / parquet  
- `optuna` → optimización de hiperparámetros  
- `shap` → interpretabilidad de modelos  
- `pmdarima` → modelos ARIMA automáticos para series temporales  

---

## 🔎 Detección de plataforma

- En **Mac con chip ARM (Apple Silicon)**, instalará `tensorflow-macos`.  
- En otros sistemas, instalará `tensorflow`.  

Esto evita problemas comunes de compatibilidad.

---

## 🐢 Notas

- Usa el mismo **intérprete de Python** con el que trabajas en tus notebooks.  
- Si usas **entornos virtuales** (recomendado), actívalos antes de correr el script.  
- Si no tienes permisos, añade `--user` a `pip` dentro del script o usa un virtualenv.  
- Tras la instalación, **reinicia tu kernel de Jupyter o terminal** para que los cambios se reconozcan.  
