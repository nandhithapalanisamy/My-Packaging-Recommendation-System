import psycopg2
import pandas as pd


# connecting to postgresql database
connection = psycopg2.connect(
    host="localhost",
    database="infosys_database",
    user="postgres",
    password="PostgreSQL#11",
    port="5432"
)

query = "SELECT * FROM target_material_data;"
df = pd.read_sql(query, connection)

print("Connected to DB")

# Save to CSV
df.to_csv("C:\InfosysInternshipRepos\Packaging-Recommendation-System\Dataset_Preparation\data_set.csv", index=False)

# Close connection
connection.close()

print("Data successfully stored in CSV file")