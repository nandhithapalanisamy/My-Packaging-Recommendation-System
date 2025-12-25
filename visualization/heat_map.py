import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"C:\InfosysInternshipRepos\Packaging-Recommendation-System\Dataset_Preparation\data_set.csv") 

# Select numeric columns
heatmap_data = df[
    ["strength", "weight_capacity", "biodegradability_score",
     "co2_emission", "recyclability", "water_resistance"]
]

# Create correlation matrix
corr_matrix = heatmap_data.corr()

# Plot heat map
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heat Map of Material Sustainability Attributes")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\correlation_heat_map.png")
plt.show()

