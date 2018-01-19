import numpy as np


mport numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# evenly sampled time at 200ms intervals
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import future        # pip install future
import builtins      # pip install future
import past          # pip install future
import six           # pip install six
# Iterations;Meilleure Fitness;Moyenne Fitness


file = open("graph1.txt", "r")

aux_line = []
y =[[] for _ in range(4)]
# Moyenne Fitness

fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.grid(True)
gridlines = ax0.get_xgridlines() + ax0.get_ygridlines()
plt.yscale('linear')
plt.xscale('linear')
print(gridlines)
for line in gridlines:
    line.set_linestyle('-.')
plt.plot(y[0], y[3], 'bs',y[0], y[3])
plt.ylabel('Distance')
plt.xlabel('Iterations')

blue_patch = mpatches.Patch(color='blue', label='Moyenne Distance')
plt.legend(handles=[blue_patch])
plt.show()