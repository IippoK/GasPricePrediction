import matplotlib.pyplot as plt
from DataParsing import get_data

plt.figure()
plt.scatter(get_data().merged["brent_price"], get_data().merged["e95_price"])
plt.xlabel("Brent Oil Price")
plt.ylabel("E95 Price")

plt.figure()
plt.scatter(get_data().merged["usd_eur"], get_data().merged["e95_price"])
plt.xlabel("USD/EUR")
plt.ylabel("E95 Price")

plt.show()
