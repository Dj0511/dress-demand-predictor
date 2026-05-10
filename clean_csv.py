import pandas as pd

# Load
df = pd.read_csv("data/sales_data.csv")

# Check columns
print("Columns:", df.columns.tolist())
print("Total rows:", len(df))
print(df.head())

# Aggregate by code
df_agg = df.groupby("code").agg(
    total_qty  = ("qty", "sum"),
    avg_rate   = ("rate", "mean"),
    num_orders = ("code", "count")
).reset_index()

df_agg["code"] = df_agg["code"].astype(str)

# Save
df_agg.to_csv("data/aggregated_sales.csv", index=False)

print("\nDone!")
print(f"Unique dress codes found: {len(df_agg)}")
print(df_agg)