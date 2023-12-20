import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

dataset = np.loadtxt('DS5.txt')

hull = ConvexHull(dataset)

plt.figure(figsize=(960/80, 540/80))

for simplex in hull.simplices:
    plt.plot(dataset[simplex, 1], dataset[simplex, 0], 'b-')

plt.scatter(dataset[:, 1], dataset[:, 0], color='red')

plt.savefig('convex_hull.png', format='png')

plt.savefig('convex_hull.png')

plt.show()
