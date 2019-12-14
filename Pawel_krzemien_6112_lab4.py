# Zadanie 1
# Poniżej przy pomocy funkcji fetch_openml można zaimportować zbiór danych o samochodach
# Zapoznaj się ze zbiorem danych (zwizualizuj wybrane przez siebie cechy) a następnie podziel go na zbiór uczący i testowy
# Wytrenuj klasyfikator KMeans z zadaną przez siebie liczbą klas i zwizualizuj wyniki predykcji na zbiorze testowym

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

# Poniżej przy pomocy funkcji fetch_openml można zaimportować zbiór danych o samochodach
cars = fetch_openml('cars1')
print('Klucze:')
print(cars.keys())
print('Nazwy:')
print(cars['feature_names'])
print('Kategorie:')
print(cars['categories'])
print(cars['data'][0])
#print(cars['DESCR'])

# Podzielmy zbiór na cechy oraz etykiety - to już znamy
# Dla uproszczenia wybieramy tylko cechę czwartą i szóstą, tj ilość koni mechanicznych i czas do 60 mil na godzinę
X = cars.data[:, [3, 5]]
y = cars['target']
y = [int(elem) for elem in y]

# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Tworzymy klasyfikator z pięcioma klastrami (klasami)
kmn = KMeans(n_clusters=5)

# Uczymy klasyfikator na danych treningowych
kmn.fit(X_train)

# Wyciągamy punkty centralne klastrów - pokażemy je na wykresie obok punktów ze zbioru uczącego
centra = kmn.cluster_centers_

fig, ax = plt.subplots(1, 2)

# pierwszy wykres to nasz zbiór uczący, z prawdziwymi klasami
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)

# Teraz używamy danych treningowych żeby sprawdzić co klasyfikator o nich myśli
y_pred_train = kmn.predict(X_train)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, s=20)

# Dokładamy na drugim wykresie centra klastrów
ax[1].scatter(centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()

# Próbujemy przewidzieć samochody dla zbioru testowego
y_pred = kmn.predict(X_test)

# Nowe samochody przewidziane przez klastrowanie
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20)

# Tak jak powyżej, wyświetlamy centra klastrów
plt.scatter(centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()

# Zadanie 2
# Opisz własnymi słowami, jakie klasy samochodów wg Ciebie znalazły się w zbiorze

# W klasach są pojazdy w zależności od mocy silnika (konie mechaniczne) na przyśpiesznie (0-60mph)
# W klasach są pojazdy z mniejszą ilością mocy oraz większym przyśpieszeniem
# Możemy stwierdzić w której klasie są samochody szybsze a w której wolniejsze
# Możemy stwierdzić w której klasie są samochody z większą mocą a w której słabsze