
# exploracion_basica.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos(ruta):
    print("📂 Cargando datos...")
    return pd.read_csv(ruta)

def explorar_df(df):
    print("\n📋 Información general:")
    print(df.info())
    print("\n📐 Dimensiones del DataFrame:", df.shape)
    print("\n🧾 Primeras filas:")
    print(df.head())
    print("\n❓ Valores nulos:")
    print(df.isnull().sum())
    print("\n📊 Estadísticas descriptivas:")
    print(df.describe())

def graficar(df, col1=None, col2=None):
    if col1:
        print(f"\n📈 Histograma de {col1}:")
        sns.histplot(df[col1], kde=True)
        plt.title(f'Distribución de {col1}')
        plt.show()

    if col1 and col2:
        print(f"\n📉 Gráfico de dispersión entre {col1} y {col2}:")
        df.plot.scatter(x=col1, y=col2)
        plt.title(f'{col1} vs {col2}')
        plt.show()

def correlacion(df):
    print("\n🔗 Matriz de correlación:")
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()

# 👇 Ejecución directa
if __name__ == "__main__":
    archivo = 'archivo.csv'  # cambia por tu ruta real
    df = cargar_datos(archivo)
    explorar_df(df)
    graficar(df, col1='col1', col2='col2')  # cambia col1 y col2 por nombres reales
    correlacion(df)
