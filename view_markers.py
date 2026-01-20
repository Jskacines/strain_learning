import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/dec9_left/dec9/left_2_7_integrated_measurement_data_20251209_16-19-11.csv")
print(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(15):
    x = data[f"marker_{i}_x (m)"]
    y = data[f"marker_{i}_y (m)"]
    z = data[f"marker_{i}_z (m)"]
    ax.plot(x,y,z)
ax.set_aspect("equal")
plt.show()



