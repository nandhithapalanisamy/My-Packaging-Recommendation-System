import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import csv data
df = pd.read_csv(r"C:\InfosysInternshipRepos\Packaging-Recommendation-System\data_source\cleaned_material_data.csv")

#material_type + recyclability bar chart
#Helps compare recyclability across different materials
plt.figure()
plt.bar(df["material_type"], df["recyclability"])
plt.xticks(rotation=90)
plt.title("Recyclability of Materials")
plt.ylabel("Recyclability (%)")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\material_type_vs_recyclability.png")

#material_type + co2_emission bar chart
#Shows which materials produce less CO₂
plt.figure()
plt.bar(df["material_type"], df["co2_emission"])
plt.xticks(rotation=90)
plt.title("CO2 Emission of Materials")
plt.ylabel("Recyclability (%)")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\material_type_vs_co2_emission.png")

#weight_capacity + co2_emission scatter plot
#Does heavier material emit more CO₂?
plt.figure()
plt.scatter(df["weight_capacity"], df["co2_emission"])
plt.xlabel("CO₂ Emission")
plt.ylabel("Recyclability (%)")
plt.title("CO₂ Emission vs Recyclability")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\weight_capacity_vs_co2_emission.png")
plt.show()

#biodegradability_score + recyclability
#Shows whether high biodegradability leads to high recyclability
plt.figure()
plt.scatter(df["biodegradability_score"], df["recyclability"])
plt.xlabel("Biodegradability Score")
plt.ylabel("Recyclability (%)")
plt.title("Biodegradability vs Recyclability")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\biodegradability_vs_recyclability.png")
plt.show()

#co2_emission
#Shows distribution of CO₂ levels
#Identifies low, medium, high emission groups
plt.figure()
plt.hist(df["co2_emission"], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of CO2 Emission")   
plt.xlabel("CO2 Emission")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\co2_emission_distribution.png")
plt.show()

#recyclability
#Understands how many materials are highly recyclable
plt.figure()
plt.hist(df["recyclability"], bins=20, color='lightgreen', edgecolor='black')
plt.title("Distribution of Recyclability")              
plt.xlabel("Recyclability (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\recyclability_distribution.png")
plt.show()

#strength + weight_capacity bar chart
#Compares material strength vs load handling
#Useful for packaging safety analysis
plt.figure()
plt.bar(df["strength"], df["weight_capacity"])
plt.xticks(rotation=90)
plt.title("Material Strength by Type")  
plt.xlabel("Strength")
plt.ylabel("Weight Capacity")
plt.tight_layout()
plt.savefig("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\visualization\\strength_vs_weight_capacity.png")
plt.show()


