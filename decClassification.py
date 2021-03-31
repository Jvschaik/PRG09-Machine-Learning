import numpy as np
import matplotlib.pyplot as plt
from machinelearningdata import Machine_Learning_Data
from sklearn.metrics import accuracy_score
from sklearn import tree

def extract_from_json_as_np_array(key, json_data):
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "0852893" # done: aanpassen aan je eigen studentnummer

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)

# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()
#print(classification_training)

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training) 
# X heeft alle coordinaten
#Y heeft alle labels
Y = extract_from_json_as_np_array("y", classification_training)

# Done: leer de classificaties
c1 = []
c2 = []
for i in range(len(classification_training)):
    if classification_training[i]['y'] == 0:
        c1.append(classification_training[i])
    if classification_training[i]['y'] == 1:
        c2.append(classification_training[i])
XC1 = extract_from_json_as_np_array("x", c1)
XC2 = extract_from_json_as_np_array("x", c2)
YC1 = extract_from_json_as_np_array("y", c1)
YC2 = extract_from_json_as_np_array("y", c2)

c1x = XC1[...,0]
c1y = XC1[...,1]
c2x = XC2[...,0]
c2y = XC2[...,1]

x = X[...,0]
y = X[...,1]

plt.axis([min(x), max(x), min(y), max(y)])

plt.scatter(c1x, c1y, s = 10, c='red')
plt.scatter(c2x, c2y, s = 10, c='blue')
# plt.scatter(x, y, s = 10)
plt.show()
decisionTree = tree.DecisionTreeClassifier().fit(X, Y)

# Done: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict
Y_predict = decisionTree.predict(X)

m, b = np.polyfit(x, Y_predict, 1)
X_plot = np.linspace([min(x), max(x)], 1)
plt.plot(X_plot, m*X_plot + b, '-')

# Done: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
treeAccuracyScore = accuracy_score(Y, Y_predict)
print("Classificatie accuratie training(Tree): " + str(treeAccuracyScore))
f = plt.figure(1)
plt.scatter(x, y, c = Y_predict, s = 10)
f.show()

# haal data op om te testen
classification_test = data.classification_test()
# Test stoppen we in variabel X_test
X_test = extract_from_json_as_np_array("x", classification_test)

# Done: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

#maken labels en coordinaten
x_test = X_test[...,0]
y_test = X_test[...,1]
# controleren of de coordinaten goed zijn met predict
Z = decisionTree.predict(X_test)
g = plt.figure(2)

#blauwe lijn, om relatief goed het gemiddelde te pakken
m, b = np.polyfit(x_test, Z, 1)
X_plot = np.linspace([min(x)], 1)
plt.plot(X_plot, m*X_plot + b, '-')

plt.scatter(x_test, y_test, c = Z, s = 10)
g.show()

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie test(Tree): " + str(classification_test))
input("Press Enter to continue...")
