import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="results_T2_gas10000.h5")
parser.add_argument("--outfile", type=str, default="results_T2_gas10000.csv")
args = parser.parse_args()

input_file = args.infile
output_file = args.outfile

df = pd.read_hdf(input_file, "data")

# Mark whether a solution was found
df["solved"] = df["solution"].notna()

# Convert reference and solution to strings for easier CSV export
df["reference"] = df["reference"].astype(str)
df["solution"] = df["solution"].astype(str)

# Select relevant columns
df = df[["reference", "solution", "solved", "nb_steps", "wall_ms"]]

df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Saved {len(df)} rows to {output_file}")
print(df.head(10))