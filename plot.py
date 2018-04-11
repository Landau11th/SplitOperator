import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('average_W_z.csv')

fig, ax = plt.subplots()

ax.semilogx(data['taus'], data['mean_W_positvie'],'ro', data['taus'],data['mean_W_negative'], 'bo')
#ax.semilogx(tau, sigma, 'x')
ax.grid()
#plt.savefig('test.png')
plt.show()