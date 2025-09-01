
#!/usr/bin/env python3

"""
crear_nb_util.py
----------------
Crea un notebook .ipynb en la ruta: ./M{modulo}/S{sesion}/
con una celda inicial que prepara el entorno (import de utilidades.tools).

Uso:
  - Ejecuta el script (doble clic si tu OS lo permite, o por terminal).
  - Ingresa el módulo (ej: 8 o M8).
  - Ingresa la sesión (ej: 5 o S5).
  - (Opcional) Ingresa el nombre de archivo o deja vacío para usar el formato "Pres s{sesion}m{modulo}.ipynb".
"""

import os
from pathlib import Path
import re
import json
from datetime import datetime

def norm_mod(value: str) -> int:
    m = re.search(r'(\d+)', value)
    if not m:
        raise ValueError("No se encontró número de módulo en la entrada.")
    return int(m.group(1))

def norm_ses(value: str) -> int:
    s = re.search(r'(\d+)', value)
    if not s:
        raise ValueError("No se encontró número de sesión en la entrada.")
    return int(s.group(1))

def make_nb_json(title: str, rel_to_root: str = "../..") -> dict:
    # Notebook básico con una celda inicial de entorno
    first_cell = f"""\
import sys, os
sys.path.insert(0, os.path.abspath("{rel_to_root}"))
from utilidades.tools import *
# from utilidades import tools_extra  # descomenta si usarás extras

# Recarga automática si editas módulos
%load_ext autoreload
%autoreload 2

print("Entorno listo: pandas, numpy, sklearn, statsmodels, etc.")
"""

    header_cell = f"""\
# {title}


"""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": header_cell
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": first_cell
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb

def main():
    print("=== Creador de notebook (desde utilidades/) ===")
    mod_in = input("Módulo (ej: 8 o M8): ").strip()
    ses_in = input("Sesión (ej: 5 o S5): ").strip()
    fname_in = input("Nombre de archivo (opcional, ej: Pres s5m8.ipynb): ").strip()

    M = norm_mod(mod_in)
    S = norm_ses(ses_in)

    root = Path(__file__).resolve().parent.parent  # raíz del proyecto (sube desde utilidades/)
    target_dir = root / f"M{M}/S{S}"
    target_dir.mkdir(parents=True, exist_ok=True)

    if fname_in:
        if not fname_in.endswith(".ipynb"):
            fname_in += ".ipynb"
        nb_path = target_dir / fname_in
    else:
        # Formato por defecto similar a tu ejemplo: "Pres s5m8.ipynb"
        nb_path = target_dir / f"Pres s{S}m{M}.ipynb"

    # Calcular ruta relativa desde el notebook hacia la raíz del proyecto
    # Si el notebook está en M*/S*, para llegar a la raíz sube 2 niveles: ../..
    rel_to_root = "../.."

    title = f"Pres S{S} M{M} — {datetime.now().strftime('%d/%m/%Y')}"
    nb = make_nb_json(title, rel_to_root=rel_to_root)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"✓ Notebook creado: {nb_path}")

if __name__ == "__main__":
    main()
