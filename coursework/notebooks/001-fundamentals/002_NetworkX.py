import networkx as nx

import warnings 
import random
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


x = []
y = []
for _ in range(1000):
    x.append(random.randint(1,1000))
    y.append(random.randint(1,1000))
print(x)

edges = list(product(x,y))

##Creamos y rellenamos nuestra grafica
G = nx.Graph()
G.add_edges_from(edges)

nx.draw(G, with_labels=True, node_color="skyblue", node_size=0.4, font_weight="bold")
plt.show()


