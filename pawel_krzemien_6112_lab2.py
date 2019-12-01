##Lab 2 6112
from sklearn import datasets

print("####################----zadanie_1----#######################")
# Zadanie 1: sprawdź poniżej inne elementy wczytanego zbioru danych, w szczególności opis.
# Opisz w max 3 zdaniach swoimi słowami co zawiera zbiór danych

iris = datasets.load_iris()
print('Opis irysów w zbiorze to: ', iris['DESCR'])
##
# Opis zawiera: 
#-Po 50 instancji 3 klas irysów (150 w sumie). 
#-Posiada ich 2 chechy: wysokość oraz szerokość. 
#-Autora oraz datę. 
#-Odnośniki do bibliografi.

print("####################----zadanie_2----#######################")
# Zadanie 2:
# Stwórz listę kilku wybranych przez siebie wartości dla parametru n_neighbors
# W pętli 'for' użyj kolejnych wartości parametru do stworzenia klasyfikatora
# Następnie naucz go na danych uczących
# Zapisz wynik scoringu na danych testowych do osobnej listy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Podzielmy zbiór na cechy oraz etykiety
# Konwencja, często spotykana w dokumentacji sklearn to X dla cech oraz y dla etykiet
X = iris.data
y = iris.target

# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
 
lista_n = [7, 13, 15, 16, 23, 29, 38, 41, 45, 58]
dokladnosci = []
 
for n_neighb in lista_n:
# W pętli 'for' użyj kolejnych wartości parametru do stworzenia klasyfikatora
  knn = KNeighborsClassifier(n_neighb)
 
# Uczymy klasyfikator na zbiorze - zaskoczenie - uczącym
  knn.fit(X_train, y_train)

# Przewidujemy wartości dla zbioru testowego
  y_pred = knn.predict(X_test)

# Sprawdzamy kilka pierwszych wartości przewidzianych
  print("5 pierwszych wartości przewidzianych")
  print(y_pred[:5])

# Sprawdzamy dokładność klasyfikatora
  print("dokładność klasyfikatora")
  print(knn.score(X_test, y_test))
 
# Zapisz wynik scoringu na danych testowych do osobnej listy
  dokladnosci.append(knn.score(X_test, y_test))

%matplotlib inline

# Tworzymy płaszczyznę wszystkich możliwych wartości dla cechy 0 oraz 2, z krokiem 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Uczymy klasyfikator na tylko dwóch wybranych cechach
knn.fit(X_train[:, [0, 2]], y_train)

# Przewidujemy każdy punkt na płaszczyźnie
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
 
# Tworzymy contourplot
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.bwr)
plt.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
plt.show()

print("Liczba sasiadow, Dokladnosc skoringu")
 
for liczba_sasiadow, dokladnosc_skoringu in zip(lista_n, dokladnosci):
    print([liczba_sasiadow, dokladnosc_skoringu])
 
# Wyświetl wykres zależności między liczbą sąsiadów a dokładnością.
plt.plot(lista_n, dokladnosci)
plt.xlabel("Liczba_n")
plt.ylabel("Dokladnosci")
plt.show()

print("####################----zadanie_3----#######################")
# Zadanie 3:
# Zbadaj zbiór danych. Stwórz wykresy obrazujące ten zbiór danych.
# Podziel zbiór danych na uczący i testowy.
# Wytrenuj klasyfikator kNN
# Dokonaj predykcji na zbiorze testowym
# Wypisz raport z uczenia: confusion_matrix oraz classification_report

# wykresy będą tworzone przy pomocy pakietu seaborn
import seaborn as sns

### wczytaj dane o winach za pomocą funkcji poniżej

from sklearn.datasets import load_wine
wine = datasets.load_wine()

### Zbadaj zbiór danych. Stwórz wykresy obrazujące ten zbiór danych.

# Zobaczmy jakie dane mamy w zbiorze
#print('Opis win w zbiorze to: ', wine['DESCR'])
print('Elementy zbioru: ', list(wine.keys()))
# Etykiety które występują
print('Cechy win w zbiorze to: ', wine['feature_names'])


# konwersja na obiekt pandas.DataFrame
wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])

# funkcja która nam zamieni wartości 0, 1, 2 na pełny opis tekstowy dla gatunku
targets = map(lambda x: wine['target_names'][x], wine['target'])

# doklejenie informacji o gatunku do reszty dataframe
wine_df['species'] = np.array(list(targets))

# wykres
sns.pairplot(wine_df, hue='species')
plt.show()

# zobaczmy jak naocznie wyglądają dane
#wine_df.head(5)


### Podziel zbiór danych na uczący i testowy.
### Wytrenuj klasyfikator kNN
### Dokonaj predykcji na zbiorze testowym

X = wine.data
y = wine.target

# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Tworzymy klasyfikator k-NN używając parametru 5 sąsiadów
knn = KNeighborsClassifier(n_neighbors = 5)

# Uczymy klasyfikator na zbiorze - zaskoczenie - uczącym
knn.fit(X_train, y_train)

# Przewidujemy wartości dla zbioru testowego
y_pred = knn.predict(X_test)

# Sprawdzamy kilka pierwszych wartości przewidzianych
print("5 pierszych wartości przewidzianych")
print(y_pred[:5])

# Sprawdzamy dokładność klasyfikatora
print("dokładność klasyfikatora:")
print(knn.score(X_test, y_test))


### Wypisz raport z uczenia: confusion_matrix oraz classification_report
from sklearn.metrics import classification_report, confusion_matrix

print("Raport z uczenia: confusion_matrix")
print(confusion_matrix(y_test, y_pred))
print("Raport z uczenia: classification_report")
print(classification_report(y_test, y_pred))

