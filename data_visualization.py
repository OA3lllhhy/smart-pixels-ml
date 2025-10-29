import pandas as pd
import matplotlib.pyplot as plt
import os

print("Loading data...")
dataX = pd.read_parquet("datasets/recon3D/recon3D_d17301.parquet")
labels_df = pd.read_parquet("datasets/labels/labels_d17301.parquet")
reshaped_dataX = dataX.values.reshape((len(dataX), 20, 13, 21))

os.makedirs("figures", exist_ok=True)

print(labels_df.iloc[0])
plt.imshow(reshaped_dataX[0, 0, :, :], cmap='coolwarm')  # first time-step
# plt.savefig("figures/first_timestep.png", dpi=300, bbox_inches='tight')
plt.close()
plt.imshow(reshaped_dataX[0, -1, :, :], cmap='coolwarm')  # last time-step
# plt.savefig("figures/last_timestep.png", dpi=300, bbox_inches='tight')
plt.close()

print("Figures saved in figures/ directory.")
