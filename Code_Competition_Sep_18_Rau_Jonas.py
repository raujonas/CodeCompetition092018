from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Einlesen der Trainings-Daten und speichern in einem Data Frame
dataset_url_train = r'C:\Users\Jonas\Downloads\verkehrsunfaelle_train.csv'
data_train = pd.read_csv(dataset_url_train, sep=',', encoding='latin-1')

#Indexspalte wird gelöscht
data_train = data_train.drop(columns='Unnamed: 0')

#Eventuell vorhandene deutsche Monatsbezeichnungen werden nach Englisch übersetzt
data_train.replace('Okt', 'Oct', inplace=True, regex=True)
data_train.replace('Mai', 'May', inplace=True, regex=True)
data_train.replace('Dez', 'Dec', inplace=True, regex=True)
data_train.replace('Mrz', 'Mar', inplace=True, regex=True)

#Funktion, um die verschiedenen Datums zuerst in ein einheitliches Format zu bringen
def try_parsing_date(text):
    for fmt in ('%d. %b.', '%d-%b-%y'):
        try:
            return time.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')
    
#Funktion, um die verschiedenen Uhrzeiten zuerst in ein einheitliches Format zu bringen
def try_parsing_time(text):
    for fmt in ('%M', '%H%M'):
        try:
            return time.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid time format found')

#Von der Unfallschwere in den Trainingsdaten wird 1 subtrahiert, damit das neuronale Netz damit rechnen kann (Klassen sind später 0,1 und 2)    
data_train['Unfallschwere'] = data_train.apply(lambda row: row['Unfallschwere']-1, axis=1)

#Unfalldatum wird geparst. Dabei habe ich mich entschieden, nur den Monat (z.B. April) zu betrachten, da bei vielen Einträgen das Jahr gefehlt hat.
#Dadurch sind 12 Kategorien enstanden, was meiner Ansicht besser ist als viele verschiedene einzelne Datums über das Jahr verteilt.    
data_train['Unfalldatum'] = data_train.apply(lambda row: try_parsing_date(row['Unfalldatum']), axis=1)
data_train['Unfalldatum'] = data_train.apply(lambda row: time.strftime("%B", (row['Unfalldatum'])), axis=1)

#Unfallzeit wird geparst. Dabei habe ich analog nur die volle Stunde betrachtet (z.B. 13 Uhr). Die Minuten zu betrachtet hätte es m.E. unnötig
#aufgebläht
data_train['Zeit (24h)'] = data_train.apply(lambda row: str(row['Zeit (24h)']), axis=1)
data_train['Zeit (24h)'] = data_train.apply(lambda row: try_parsing_time(row['Zeit (24h)']), axis=1)
data_train['Zeit (24h)'] = data_train.apply(lambda row: time.strftime("%H", (row['Zeit (24h)'])), axis=1)
data_train['Zeit (24h)'] = data_train.apply(lambda row: int(row['Zeit (24h)']), axis=1)

#Hier werden die Spalten definiert, die kein Integer-Format haben und klassifiziert werden müssen. Dazu nutze ich get_dummies von pandas
cols_to_transform = ['Strassenklasse','Unfalldatum','Unfallklasse','Lichtverhältnisse','Bodenbeschaffenheit','Geschlecht','Fahrzeugtyp','Wetterlage']
data_train = pd.get_dummies(data_train, columns=cols_to_transform)

#Aufteilung der Trainingdaten im Verhältnis 10:1, um Training des Netzes danach validieren zu können
data_train, data_validate = train_test_split(data_train, test_size=0.1)

#Die Anzahl der Zeilen und der Features wird in eine temporäre CSV-Datei geschrieben
num_train_entries = data_train.shape[0]
num_train_features = data_train.shape[1]-1
num_validate_entries = data_validate.shape[0]
num_validate_features = data_validate.shape[1]-1
data_train.to_csv('train_temp.csv', index=False)
data_validate.to_csv('validate_temp.csv', index=False)
open('data_train.csv', 'w').write(str(num_train_entries) +
                                      "," + str(num_train_features) +
                                      "," + open('train_temp.csv').read())

open('data_validate.csv', 'w').write(str(num_validate_entries) +
                                     "," + str(num_validate_features) +
                                     "," + open("validate_temp.csv").read())

#Hier wird jeweils ein Set zum Training und zur Validierung erstellt. Target_column gibt die Spalte an, die das Netz vorhersagen soll
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='data_train.csv',
    target_dtype=np.int,
    features_dtype=np.int,
    target_column=1)

validating_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename='data_validate.csv',
    target_dtype=np.int,
    features_dtype=np.int,
    target_column=1)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=65)]

#Erstellung des classifiers, der zwischen 3 Klassen unterscheiden soll
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3)

#Geben die Test- und Validierungs-Daten zurück
def train_inputs():
    a = tf.constant(training_set.data)
    b = tf.constant(training_set.target)
    return a,b

def validate_inputs():
    a = tf.constant(validating_set.data)
    b = tf.constant(validating_set.target)
    return a,b

#Das Model wird gefittet
classifier.fit(input_fn=train_inputs, steps=2000)

#Validierung mit den kleineren Validierungsdatensatz, der oben getrennt wurde (ergab in der Spitze über 90%)
accuracy_score = classifier.evaluate(input_fn=validate_inputs, steps=1)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

###############################################################################

#Ab diesem Schritt werden die Test-Daten eingelesen und bearbeitet. Die nachfolgenden Schritte zur Bearbeitung der Daten erfolgen wie 
#bereits im oberen Abschnitt beschrieben
dataset_url_test = (r'C:\Users\Jonas\Downloads\verkehrsunfaelle_test.csv')
data_test = pd.read_csv(dataset_url_test, sep=',', encoding='latin-1')

data_test = data_test.drop(columns='Unnamed: 0')

data_test.replace('Okt', 'Oct', inplace=True, regex=True)
data_test.replace('Mai', 'May', inplace=True, regex=True)
data_test.replace('Dez', 'Dec', inplace=True, regex=True)
data_test.replace('Mrz', 'Mar', inplace=True, regex=True)
        
data_test['Unfalldatum'] = data_test.apply(lambda row: try_parsing_date(row['Unfalldatum']), axis=1)
data_test['Unfalldatum'] = data_test.apply(lambda row: time.strftime("%B", (row['Unfalldatum'])), axis=1)

data_test['Zeit (24h)'] = data_test.apply(lambda row: str(row['Zeit (24h)']), axis=1)
data_test['Zeit (24h)'] = data_test.apply(lambda row: try_parsing_time(row['Zeit (24h)']), axis=1)
data_test['Zeit (24h)'] = data_test.apply(lambda row: time.strftime("%H", (row['Zeit (24h)'])), axis=1)
data_test['Zeit (24h)'] = data_test.apply(lambda row: int(row['Zeit (24h)']), axis=1)

data_test = pd.get_dummies(data_test, columns=cols_to_transform)

#Bei der Prediction ergaben sich Probleme, da weniger Vielfalt an Attribut-Ausprägungen als im Training-Datensatz vorhanden waren.
#Deshalb erzeugte die pd.get_dummies-Funktion weniger Spalten, weshalb nun so lange Spalten hinzugefügt werden, bis die Anzahl der Spalten
#im Training-Data-Frame erreicht wurde (Da get_dummies für jedes Attribut eine neue Spalte erzeugt). Somit matcht das Modell mit dem 
#oben erstellen classifier wieder und kann benutzt werden
while len(data_test.columns) < len(data_train.columns)-1:
    print(len(data_test.columns))
    data_test[len(data_test.columns)] = 0
array_test = np.array(data_test)

#Hier wird die prediction durchgeführt
prediction = classifier.predict(np.array(data_test, dtype=float), as_iterable=False)

#Die predection wird in einem CSV-File auf der Festplatte gespeichert.
data_prediction = pd.DataFrame()
data_test = pd.read_csv(dataset_url_test, sep=',', encoding='latin-1')
data_prediction['Unfall_ID'] = data_test['Unnamed: 0']
data_prediction['Unfallschwere'] = prediction
data_prediction['Unfallschwere'] = data_prediction.apply(lambda row: row['Unfallschwere']+1, axis=1)

data_prediction.to_csv('data_prediction.csv', index=False)
