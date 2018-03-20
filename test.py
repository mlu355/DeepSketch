import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (100,100)

# confmat=np.random.rand(250,250)
confmat = np.load("conf_matrix.npy")
print(confmat.shape)
ticks=np.arange(250)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()