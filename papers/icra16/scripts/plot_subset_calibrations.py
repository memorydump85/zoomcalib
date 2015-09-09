#! /usr/bin/python

"""
Plots the output of subset_calibrations.py
"""

import fileinput
data = []
for line in fileinput.input():
    data.append([ float(v) for v in line.split() ])

import numpy as np
data = np.vstack(data)
print data.shape

from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(8,5))

plt.subplot(121)
plt.hist(np.hstack([data[:,0], data[:,1]]), bins=50)

plt.subplot(122)
plt.plot(data[:,2], data[:,3], '.', alpha=0.5)
plt.axis('equal')

plt.show()