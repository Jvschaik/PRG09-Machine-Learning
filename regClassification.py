from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)


# Done: leer de classificaties
x = X[...,0]
y = X[...,1]

plt.axis([min(x), max(x), min(y), max(y)])
regression = LogisticRegression().fit(X, Y)

# Done: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict
Y_predict = regression.predict(X)

m, b = np.polyfit(x, Y_predict, 1)
X_plot = np.linspace([min(x)], 1)
plt.plot(X_plot, m*X_plot + b, '-')

regAccuracyScore = accuracy_score(Y, Y_predict)
print("Classificatie accuratie training(Reg): " + str(regAccuracyScore))

# Done: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
plt.scatter(x, y, c = Y_predict, s = 10)
plt.show()

# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# Done: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.
x_test = X_test[...,0]
y_test = X_test[...,1]
# Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?
Z = regression.predict(X_test)

m, b = np.polyfit(x_test, Z, 1)
X_plot = np.linspace([min(x_test)], 1)
plt.plot(X_plot, m*X_plot + b, '-')

plt.scatter(x_test, y_test, c = Z, s = 10)
plt.show() 
# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt

# tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
classification_test = data.classification_test(Z.tolist()) 
print("Classificatie accuratie test(Reg): " + str(classification_test))
