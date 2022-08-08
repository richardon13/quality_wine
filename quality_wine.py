from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier


# Importar DB

url_red = 'https://raw.githubusercontent.com/terranigmark/curso-analisis-exploratorio-datos-platzi/main/winequality-red.csv'
url_white = 'https://raw.githubusercontent.com/terranigmark/curso-analisis-exploratorio-datos-platzi/main/winequality-white.csv'

wine_red = pd.read_csv(url_red, sep=';')
wine_white = pd.read_csv(url_white, sep=';')

wine_white.head()

# Forma 1 de unir los DF red & white

wine_red['color'] = 'red'
wine_white['color'] = 'white'


df_wine = wine_red.append(wine_white)
df_wine
print('='*158)

# Forma 2 de unir los DF red & white

df_wine = pd.concat([wine_red, wine_white])
print(df_wine.head())
print('='*158)

# Determinamos los valores estadisticos de algunas de las variables de forma transpuesta

print(df_wine.describe().T)
print('='*158)

# Ploteo los valores
df_wine.plot()
plt.show()

# Grafico la densidad

df_wine['density'].plot()
plt.show()

#R/= Descubro que la densidad posee valores atipicos 

# Descubro cuales son valores que determinan la calidad del vino y los grafico
sns.set(rc={'figure.figsize': (14, 8)})
sns.countplot(df_wine['quality'])
plt.show()

#R/ = Por indagacion previa, se descubrio que mientras mas alto el valor mejor es la calidad del vino.
# Por eso divido la calidad del vino en sus 3 mayores valores que corresponden a 
# Baja <= 5, Media = 6, Alta => 7

# Hago una visualizacion grafica de los patrones. Se observan las distribuciones de las mismas
# sns.pairplot(df_wine)
# plt.show()

# Grafico la correlacion entre las variables

sns.heatmap(df_wine.corr(), annot = True, fmt = '.2f', linewidths = 2, cmap = 'coolwarm')
plt.show()

# Debido a la alta correlacion que existe entre el alcohol y la densidad, procedemos a hacer su 
# respectiva grafica comparativa

sns.distplot(df_wine['alcohol'])
plt.show()

# R/= El alcohol es muy importante para determinar la calidad del vino debido a su alta correlacion

sns.boxplot(x='quality', y='alcohol', data=df_wine)
plt.show()

#R/= Se observa en el grafico de cajas que la calidad del vino 5 posee una cantidad considerable de 
# datos atipicos (outliers)

df_wine['quality_label'] = df_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
print(df_wine)
print('='*158)

# Convierto la valriable 'quality_label en categorica'

df_wine['quality_label'] = pd.Categorical(df_wine['quality_label'], categories=['low', 'medium', 'high'])
print(df_wine)
print(df_wine.dtypes)
print('='*158)

from mpl_toolkits.mplot3d import Axes3D

fig = df_wine.hist(bins=15, color='b', edgecolor='darkmagenta', linewidth=1.0, xlabelsize=10, ylabelsize=10, xrot=45, yrot=0, figsize=(8,7), grid=False)
plt.tight_layout(rect=(0, 0, 1.5, 1.5))
plt.show()

# Analisis de Regresion con Scikit-Learn 

# Convierto la variable quality_label de categorica string en categorica numerica

label_quality = LabelEncoder()
df_wine['quality_label'] = label_quality.fit_transform(df_wine['quality_label'])
print(df_wine['quality_label'].unique())
print('='*158)

# R/= Se convirtieron los valores de la variable quality_label en numerica. se puede observar que low=1, medium=2, high=0

# Solo trabajo con las columnas de interes

df_wine_training = df_wine.drop(['color', 'quality_label'], axis=1)
X = df_wine_training.values
y = df_wine['quality_label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .30, random_state=42)

# Aplicando Regresion Logistica

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Exactitud de:', sklearn.metrics.accuracy_score(y_test, y_pred))
# R/= Con LogisticRegression() Se alcanzo una exactitud del 94%

# Aplicando KNearestNeighbors

#model_names=['KNearestNeighbors']

#acc=[]
#eval_acc={}
#classification_model=KNeighborsClassifier()
#classification_model.fit(X_train,y_train)
#pred=classification_model.predict(X_test)
#acc.append(accuracy_score(pred,y_test))
#eval_acc={'Modelling Algorithm':model_names,'Accuracy':acc} 

#print('Exactitud de:', eval_acc)

# R/= Con KNearestNeighbors se alcanzo una exactitud del 71%
