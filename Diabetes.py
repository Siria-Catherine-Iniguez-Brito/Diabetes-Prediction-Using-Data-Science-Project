#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:29:26 2025

@author: cati
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as ptl 
import seaborn as sns
from pandas.plotting import scatter_matrix



nombres =  ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv('pima-indians-diabetes.csv', names = nombres)

 
array = data.values 
X = array[:,0:8]
Y = array[:,8]


data.head(20)
print(data.shape) 
print(data.dtypes)

print(data.describe())
print(data.groupby('class').size())

correlacion = data.corr(method = 'pearson')
print(correlacion)

fig =ptl.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlacion,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks =np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(nombres)
ax.set_yticklabels(nombres)
ptl.show()

 
fig =ptl.figure(figsize=(10,10))
ptl.figure(figsize=(10,10))
ax = sns.heatmap(correlacion,vmax=1,square =True, annot=True,cmap='viridis')
ptl.title('Matriz de Correlación')



print("")
print(data.skew())

fig = ptl.figure(figsize=(10,10))
ax = fig.gca()
data.hist(ax=ax)
ptl.show()


f,axes = ptl.subplots(3,3,figsize =(14,14))
sns.distplot(data['preg'], ax =axes[0,0])
sns.distplot(data['plas'], ax =axes[0,1])
sns.distplot(data['pres'], ax =axes[0,2])
sns.distplot(data['skin'], ax =axes[1,0])
sns.distplot(data['test'], ax =axes[1,1])
sns.distplot(data['mass'], ax =axes[1,2])
sns.distplot(data['pedi'], ax =axes[2,0])
sns.distplot(data['age'], ax =axes[2,1])
sns.distplot(data['class'], ax =axes[2,2])


fig = ptl.figure(figsize= (16,16))
ax =fig.gca()
data.plot(ax=ax,kind='density', subplots=True, layout =(3,3),sharex = False)
ptl.show()


fig = ptl.figure(figsize= (16,16))
ax =fig.gca()
data.plot(ax=ax,kind='box', subplots=True, layout =(3,3),sharex = False)
ptl.show()


f,axes = ptl.subplots(3,3,figsize =(14,14))
sns.boxplot(data['preg'], ax =axes[0,0])
sns.boxplot(data['plas'], ax =axes[0,1])
sns.boxplot(data['pres'], ax =axes[0,2])
sns.boxplot(data['skin'], ax =axes[1,0])
sns.boxplot(data['test'], ax =axes[1,1])
sns.boxplot(data['mass'], ax =axes[1,2])
sns.boxplot(data['pedi'], ax =axes[2,0])
sns.boxplot(data['age'], ax =axes[2,1])
sns.boxplot(data['class'], ax =axes[2,2])


ptl.rcParams['figure.figsize']=[20,15]
scatter_matrix(data)
ptl.show()

sns.pairplot(data)
sns.pairplot(data, hue="class", diag_kind ='hist')


#VAMOS A QUITAR VALORES CORRUPTOS: 
# LA PRESION NO PUEDE SER CERO O NEGATIVA 

indices1 = data[data["pres"]<= 0].index
data = data.drop(indices1)
data = data.reset_index(drop=True)
indices2 = data[data["age"]> 110].index
data = data.drop(indices2)
data = data.reset_index(drop=True)
indices3 = data[data["mass"]<= 0].index
data = data.drop(indices3)
data = data.reset_index(drop=True)

#data = data[ data["pres"]> 0]
#data = data[ data["pres"]< 110]

#TRASFORMACIONES DE BOX-COX

from sklearn.preprocessing import PowerTransformer
features = data[['pres','age']]
pt = PowerTransformer(method='box-cox',standardize = True)


skl_boxcox = pt.fit(features)
skl_boxcox = pt.transform(features)
df_features = pd.DataFrame(data = skl_boxcox,columns =['pres','age'] )
     
data.drop(['pres'], axis = 1, inplace = True)   
data.drop(['age'], axis = 1, inplace = True) 
    
df_data = pd.concat([data,df_features],axis = 1)
cols = df_data.columns.tolist()



features2 = df_data[['test','pedi']]

print(features2)


pt = PowerTransformer(method='yeo-johnson',standardize = True)


skl_boxcox = pt.fit(features2)
skl_boxcox = pt.transform(features2)
df_features2 = pd.DataFrame(data = skl_boxcox, columns =['test','pedi'] )
     

df_data.drop(['test'], axis = 1, inplace = True)   
df_data.drop(['pedi'], axis = 1, inplace = True) 

   
df_data2 = pd.concat([df_data, df_features2],axis = 1)


cols = df_data2.columns.tolist()

cols = cols[-1:] + cols[:-1]

cols = cols[-1:] + cols[:-1]

cols = cols[-1:] + cols[:-1]

cols = cols[-1:] + cols[:-1]

df_data2 = df_data2[cols]

print('DF_DATA')
print(df_data2)

        
fig = ptl.figure(figsize=(10,10))
ax = fig.gca()
df_data2.hist(ax=ax)
ptl.show()

f,axes = ptl.subplots(3,3,figsize =(14,14))
sns.distplot(df_data2['preg'], ax =axes[0,0])
sns.distplot(df_data2['plas'], ax =axes[0,1])
sns.distplot(df_data2['pres'], ax =axes[0,2])
sns.distplot(df_data2['skin'], ax =axes[1,0])
sns.distplot(df_data2['test'], ax =axes[1,1])
sns.distplot(df_data2['mass'], ax =axes[1,2])
sns.distplot(df_data2['pedi'], ax =axes[2,0])
sns.distplot(df_data2['age'], ax =axes[2,1])
sns.distplot(df_data2['class'], ax =axes[2,2])
    


#vamos a estandarizar todos los datos
array = df_data2.values 
X = array[:,0:8]
Y = array[:,8]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
rescaledX =scaler.transform(X)

nombres =  ['pres','age','test','pedi','preg','plas','skin','mass']
df_dataStandarizacion = pd.DataFrame(rescaledX, columns = nombres)
df_dataStandarizacion['class'] = Y
print("")
print('          Conjunto de datos tras la Estandarización')
print("")
print(df_dataStandarizacion)

array = df_dataStandarizacion.values 
X1 = array[:,0:8]
Y1 = array[:,8]


# vamos a realizar una normalizacion de los datos 
array = df_data2.values 
X2 = array[:,0:8]
Y2 = array[:,8]

from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X2)
normalizerX =scaler.transform(X2)
nombres =  ['pres','age','test','pedi','preg','plas','skin','mass']
df_dataNormalizacion = pd.DataFrame(normalizerX, columns = nombres)
df_dataNormalizacion['class'] = Y

print("")
print('          Conjunto de datos tras la Normalización')
print("")
print(df_dataNormalizacion)

array = df_dataNormalizacion.values 
W1 = array[:,0:8]
Z1 = array[:,8]


f,axes = ptl.subplots(3,3,figsize =(14,14))
sns.distplot(df_dataNormalizacion['preg'], ax =axes[0,0])
sns.distplot(df_dataNormalizacion['plas'], ax =axes[0,1])
sns.distplot(df_dataNormalizacion['pres'], ax =axes[0,2])
sns.distplot(df_dataNormalizacion['skin'], ax =axes[1,0])
sns.distplot(df_dataNormalizacion['test'], ax =axes[1,1])
sns.distplot(df_dataNormalizacion['mass'], ax =axes[1,2])
sns.distplot(df_dataNormalizacion['pedi'], ax =axes[2,0])
sns.distplot(df_dataNormalizacion['age'], ax =axes[2,1])
sns.distplot(df_dataNormalizacion['class'], ax =axes[2,2])
  

"""
#ELIMINACION RFE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
selector = RFE(model,n_features_to_select=5)

X_selected = selector.fit(normalizerX,Y)
print(X_selected.support_)
print(X_selected.ranking_)
print("")
print('Conjunto de datos despues de la eliminación cn RFE')
print("")
"""




#REDUCCION DE DIMENSIONES PCA
from sklearn.decomposition import PCA 
k = 5 
pca = PCA(n_components=  k)
fit = pca.fit(X)
X_transform = pca.transform(X)
C = pca.components_

print(f"Varianza explicada : {fit.explained_variance_ratio_}")
print(f"Componets:{C}")
df_pca = pd.DataFrame(data= X_transform, columns =['P1','P2','P3','P4','P5'])


df_pca['class'] = Y
df_pca['class'] = df_pca['class'].astype(int)


print("")
print('Conjunto de datos tras la reducción de dimensiones con PCA')
print("")
print(df_pca)

array = df_pca.values 
M1 = array[:,0:5]
N1 = array[:,5]

#MODELADO 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


models = []
models.append(('LoR', LogisticRegression(solver ='lbfgs', max_iter = 1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('k-NN', KNeighborsClassifier()))
models.append(('cart', DecisionTreeClassifier()))
models.append(('NB', GaussianNB() ))
models.append(('SVM', SVC( gamma = 'auto', C = 2, kernel= 'linear')))

results1 = []
results2 = []
results3 = []
names = []

scoring = 'accuracy'

print(" ")
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 7,shuffle=True)
    cv_results1 = cross_val_score(model, X1, Y1, scoring=scoring)
    cv_results2 = cross_val_score(model, W1, Z1, scoring=scoring)
    cv_results3 = cross_val_score(model, M1, N1, scoring=scoring)
    results1.append(cv_results1)
    results2.append(cv_results2)
    results3.append(cv_results3)
    names.append(name)
    print(f"{name}: {cv_results1.mean()*100.0:,.2f}% ({cv_results1.std()*100.0:,.2f}%)")
    print(f"{name}: {cv_results2.mean()*100.0:,.2f}% ({cv_results2.std()*100.0:,.2f}%)")
    print(f"{name}: {cv_results3.mean()*100.0:,.2f}% ({cv_results3.std()*100.0:,.2f}%)")
    
    
    
fig = plt.figure()
fig.suptitle("Comparación de algoritmos")
ax = fig.add_subplot(111)
plt.boxplot(results1)
ax.set_xticklabels(names)
plt.show()
    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results2)
ax.set_xticklabels(names)
plt.show()


fig = plt.figure()
fig.suptitle("Comparación de algoritmos")
ax = fig.add_subplot(111)
plt.boxplot(results3)
ax.set_xticklabels(names)
plt.show()


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

results1 = []
results2 = []
results3 = []
names = []

print("")
print('roc_auc')
print("")
scoring = 'roc_auc'

for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 7,shuffle=True)
    cv_results1 = cross_val_score(model, X1, Y1, scoring=scoring)
    cv_results2 = cross_val_score(model, W1, Z1, scoring=scoring)
    cv_results3 = cross_val_score(model, M1, N1, scoring=scoring)
    results1.append(cv_results1)
    results2.append(cv_results2)
    results3.append(cv_results3)
    names.append(name)
    print(f"{name}: {cv_results1.mean():,.2f} ({cv_results1.std():,.2})")
    print(f"{name}: {cv_results2.mean():,.2f} ({cv_results2.std():,.2})")
    print(f"{name}: {cv_results3.mean():,.2f} ({cv_results3.std():,.2})")

resultados = results1 + results2+ results3 
names = names + names + names 
fig = plt.figure()
#fig.suptitle("Comparación de algoritmos")
ax = fig.add_subplot(111)
plt.boxplot(resultados)
ax.set_xticklabels(names)
plt.show()

"""
kfold = KFold(n_splits = 10, random_state = 7,shuffle=True)
model = LogisticRegression(solver ='lbfgs', max_iter=1000)
scoring = 'roc_auc'
results = cross_val_score(model, X1, Y1, scoring=scoring)
print(f"AUC: {results.mean()} ({results.std()})")
"""



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size= test_size,random_state= seed)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matriz = confusion_matrix(Y_test, predicted)
print(matriz)


from sklearn.metrics import classification_report
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size= test_size,random_state= seed)
model = SVC( gamma = 'auto', C = 2, kernel= 'linear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
reporte = classification_report(Y_test, predicted)
print(reporte)


from sklearn.metrics import classification_report
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size= test_size,random_state= seed)
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
reporte = classification_report(Y_test, predicted)
print(reporte)


from sklearn.metrics import classification_report
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(M1,N1,test_size= test_size,random_state= seed)
model = SVC( gamma = 'auto', C = 2, kernel= 'linear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
reporte = classification_report(Y_test, predicted)
print(reporte)




    