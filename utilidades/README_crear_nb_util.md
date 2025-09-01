# 📓 crear_nb_util.py — Generador de Notebooks

Este script crea notebooks `.ipynb` listos para trabajar con tu entorno de ciencia de datos, agregando automáticamente un **chunk inicial de imports** y configuración.

---

## 🛠️ Uso

Ejecuta el script desde la terminal:

```bash
python utilidades/crear_nb_util.py
```

El programa te hará una serie de preguntas:

1. **¿Dónde guardar el notebook?**
   - `1` → En la carpeta actual.  
   - `2` → En la estructura `M{mod}/S{ses}` (por defecto).

2. **Módulo**  
   - Ejemplo: `8` o `M8`.

3. **Sesión**  
   - Ejemplo: `5` o `S5`.

4. **Nombre del archivo** (opcional)  
   - Si lo dejas vacío, se usará el formato por defecto:  
     `Pres s{sesion}m{modulo}.ipynb`.

5. **Modo de imports**
   - `1` → Usar `utilidades.tools` (recomendado, más limpio y centralizado).  
   - `2` → Copiar el contenido actual de `tools.py` dentro del notebook (para tener los imports explícitos).  

---

## 📂 Ejemplo

Si eliges:
- Módulo = `8`  
- Sesión = `5`  
- Nombre vacío  
- Opción por defecto de carpeta (2)  

Se creará el archivo:

```
Ciencia_de_datos/M8/S5/Pres s5m8.ipynb
```

Con la primera celda ya configurada:

```python
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
from utilidades.tools import *
# from utilidades import tools_extra  # descomenta si usarás extras

%load_ext autoreload
%autoreload 2

print("Entorno listo (utilidades.tools): pandas, numpy, sklearn, statsmodels, etc.")
```

*(si elegiste opción 2 de imports, se insertará el contenido de `tools.py` en su lugar).*

---

## 🔎 Características adicionales
- **Apertura automática**: al finalizar, el script intenta abrir el notebook recién creado (compatible con Windows, macOS y Linux).  
- **Autoreload activado**: incluye `%autoreload 2`, que recarga los módulos automáticamente si modificas `tools.py` mientras trabajas.  
- **Flexible**: puedes usarlo tanto para una estructura organizada por módulos/sesiones como para notebooks sueltos en cualquier carpeta.  
