import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(6*1.5,4))

x = np.arange(-0.2,5.2,0.01)
t = (x-1.35)**3 + np.sin(2*x) + 0.2*np.random.randn(len(x))

plt.axvspan(x[230], x[250], facecolor='#cdcdcd', edgecolor='#cacaca')
plt.text(x[237], -2, 'W')

# local regression
regressor = []
for pos in xrange(20, len(x)-20):
    subset = slice(pos-20, pos+20)
    line = np.polyfit(x[subset], t[subset], deg=1)
    regressor.append(np.polyval(line, x[pos]))

plt.plot(x[20:-20], regressor, '-', color='#396AB1', linewidth=3)

plt.plot(x, t, 'k+')

subset = slice(120,140)
plt.plot(x[subset], t[subset], 'o', color='#CC2529', markersize=4)

line = np.polyfit(x[subset], t[subset], deg=1)
a, b = x[70], x[190]
plt.plot([a, b], np.polyval(line, [a, b]), '-', color='#CC2529', linewidth=1.5)

subset = slice(230,250)
plt.plot(x[subset], t[subset], 'o', color='#CC2529', markersize=4)

line = np.polyfit(x[subset], t[subset], deg=1)
a, b = x[180], x[300]
plt.plot([a, b], np.polyval(line, [a, b]), '-', color='#CC2529', linewidth=1.5)

plt.xlim([0, 3])
plt.ylim([-3, 5])
plt.tight_layout()
plt.show()
