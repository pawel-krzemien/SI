import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

currency_1 = input("Kod waluty: (AAA)")
currency_2 = input("Kod waluty: (AAA)")
date_start = input("Data początku: (rrrr-mm-dd)")
date_end = input("Data końca: (rrrr-mm-dd)")

# pkt. 1 - funkcja pzyjmująca parametr kodu waluty oraz datę początkową i końcową.
def req_currency(currency,date_start,date_end):
    currency_res = requests.get('http://api.nbp.pl/api/exchangerates/rates/A/' + currency + '/' + date_start + '/' + date_end + '/').json()
    return currency_res['rates']

# pkt. 2 - wywołanie wcześniejszej funkcji dla dwóch walut.
currency_data_1 = req_currency(currency_1,date_start,date_end)
currency_data_2 = req_currency(currency_2,date_start,date_end)
rate_dataframe_1 = pd.DataFrame.from_dict(currency_data_1).head()
rate_dataframe_2 = pd.DataFrame.from_dict(currency_data_2).head()

# pkt. 3 - Ustawienie indeksów na datę.
plot_data_1 = rate_dataframe_1.set_index(['effectiveDate'])['mid']
plot_data_2 = rate_dataframe_2.set_index(['effectiveDate'])['mid']

# pkt. 4 - korelacja kwóch kursów walut.
correlation = np.corrcoef (plot_data_1, plot_data_2)[0][1]
print('korelacja {} do {} = {}'.format(currency_1, currency_2, correlation))

# pkt.5 - wykres rysowany z wczesniej wczytanych kodów waluty.
plt.plot(plot_data_1, 'b--', plot_data_2,'r--')
plt.title('Wykres wybranych walut')
plt.ylabel('Kwota w złotówkach')
plt.xlabel('Data')
plt.ylim(ymin=0)
plt.legend([currency_1, currency_2], loc='lower left')
plt.show()