import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# enero
enero = pd.read_csv("./datos/1_Enero2023.csv", encoding="ISO-8859-1", sep=";")

enero.fillna(0, inplace=True)

suma_facturacion_enero = enero["FACTURACIÓN"].sum()

# febrero
febrero = pd.read_csv("./datos/2_Febrero2023.csv",
                      encoding="ISO-8859-1",
                      sep=";")

febrero.fillna(0, inplace=True)

suma_facturacion_febrero = febrero["FACTURACIÓN"].sum()

# marzo

marzo = pd.read_csv("./datos/3_Marzo2023.csv", encoding="ISO-8859-1", sep=";")

marzo.fillna(0, inplace=True)

suma_facturacion_marzo = marzo["FACTURACIÓN"].sum()

# abril

abril = pd.read_csv("./datos/4_Abril2023.csv", encoding="ISO-8859-1", sep=";")

abril.fillna(0, inplace=True)

suma_facturacion_abril = abril["FACTURACIÓN"].sum()

# mayo

mayo = pd.read_csv("./datos/5_Mayo2023.csv", encoding="ISO-8859-1", sep=";")

mayo.fillna(0, inplace=True)

suma_facturacion_mayo = mayo["FACTURACIÓN"].sum()

data = {
    "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo"],
    "Suma de Facturación": [
        suma_facturacion_enero,
        suma_facturacion_febrero,
        suma_facturacion_marzo,
        suma_facturacion_abril,
        suma_facturacion_mayo,
    ],
}

# Crear un DataFrame a partir del diccionario
tabla = pd.DataFrame(data)

# Agregar una columna con el número de mes
tabla["Periodo"] = range(1, len(tabla) + 1)

# Reordenar las columnas
tabla = tabla[["Mes", "Periodo", "Suma de Facturación"]]

# Mostrar la tabla
print(tabla)

# medias
x_promedio = np.mean(tabla["Periodo"])
y_promedio = np.mean(tabla["Suma de Facturación"])

print(x_promedio)
print(y_promedio)

# pendiente

tabla = pd.DataFrame(data)

# Agregar una columna con el número de mes
tabla["Periodo"] = range(1, len(tabla) + 1)

# Ajustar un modelo de regresión lineal
X = tabla["Periodo"]  # Variable independiente
X = sm.add_constant(X)  # Agregar una columna de unos para el término constante
y = tabla["Suma de Facturación"]  # Variable dependiente

model = sm.OLS(y, X)
results = model.fit()

# Obtener la pendiente
pendiente = results.params[1]

print("La pendiente (coeficiente de regresión) es:", pendiente)

# obteniendo a

a = y_promedio - pendiente * x_promedio

print(a)

# obteniendo y

periodo_predecir = 6

y = a + pendiente * periodo_predecir

print(y)

# grafico

meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "junio"]
suma_facturacion = [
    25946258.8,
    23283496.4,
    21899917.4,
    24646506.2,
    24075024.5,
    23256403.0,
]

# Trazar gráfico
plt.plot(meses, suma_facturacion, marker="o", linestyle="-", color="b")

# Etiquetas de los ejes
plt.xlabel("Mes")
plt.ylabel("Suma de Facturación")

# Límites del eje y
plt.ylim(1000000, 50000000)

# Título del gráfico
plt.title("Suma de Facturación por Mes")

# Mostrar gráfico
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# enero
enero = pd.read_csv("./datos/1_Enero2023.csv", encoding="ISO-8859-1", sep=";")

enero.fillna(0, inplace=True)

suma_facturacion_enero = enero["FACTURACIÓN"].sum()

# febrero
febrero = pd.read_csv("./datos/2_Febrero2023.csv",
                      encoding="ISO-8859-1",
                      sep=";")

febrero.fillna(0, inplace=True)

suma_facturacion_febrero = febrero["FACTURACIÓN"].sum()

# marzo

marzo = pd.read_csv("./datos/3_Marzo2023.csv", encoding="ISO-8859-1", sep=";")

marzo.fillna(0, inplace=True)

suma_facturacion_marzo = marzo["FACTURACIÓN"].sum()

# abril

abril = pd.read_csv("./datos/4_Abril2023.csv", encoding="ISO-8859-1", sep=";")

abril.fillna(0, inplace=True)

suma_facturacion_abril = abril["FACTURACIÓN"].sum()

# mayo

mayo = pd.read_csv("./datos/5_Mayo2023.csv", encoding="ISO-8859-1", sep=";")

mayo.fillna(0, inplace=True)

suma_facturacion_mayo = mayo["FACTURACIÓN"].sum()

data = {
    "Mes": ["Enero", "Febrero", "Marzo", "Abril", "Mayo"],
    "Suma de Facturación": [
        suma_facturacion_enero,
        suma_facturacion_febrero,
        suma_facturacion_marzo,
        suma_facturacion_abril,
        suma_facturacion_mayo,
    ],
}

# Crear un DataFrame a partir del diccionario
tabla = pd.DataFrame(data)

# Agregar una columna con el número de mes
tabla["Periodo"] = range(1, len(tabla) + 1)

# Reordenar las columnas
tabla = tabla[["Mes", "Periodo", "Suma de Facturación"]]

# Mostrar la tabla
print(tabla)

# medias
x_promedio = np.mean(tabla["Periodo"])
y_promedio = np.mean(tabla["Suma de Facturación"])

print(x_promedio)
print(y_promedio)

# pendiente

tabla = pd.DataFrame(data)

# Agregar una columna con el número de mes
tabla["Periodo"] = range(1, len(tabla) + 1)

# Ajustar un modelo de regresión lineal
X = tabla["Periodo"]  # Variable independiente
X = sm.add_constant(X)  # Agregar una columna de unos para el término constante
y = tabla["Suma de Facturación"]  # Variable dependiente

model = sm.OLS(y, X)
results = model.fit()

# Obtener la pendiente
pendiente = results.params[1]

print("La pendiente (coeficiente de regresión) es:", pendiente)

# obteniendo a

a = y_promedio - pendiente * x_promedio

print(a)

# obteniendo y

periodo_predecir = 6

y = a + pendiente * periodo_predecir

print(y)

# grafico

meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "junio"]
suma_facturacion = [
    25946258.8,
    23283496.4,
    21899917.4,
    24646506.2,
    24075024.5,
    23256403.0,
]

# Trazar gráfico
plt.plot(meses, suma_facturacion, marker="o", linestyle="-", color="b")

# Etiquetas de los ejes
plt.xlabel("Mes")
plt.ylabel("Suma de Facturación")

# Límites del eje y
plt.ylim(1000000, 50000000)

# Título del gráfico
plt.title("Suma de Facturación por Mes")

# Mostrar gráfico
plt.show()
