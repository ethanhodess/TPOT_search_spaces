import pandas as pd
import glob

# folder passed as argument
path = "/common/hodesse/hpc_test/TPOT_search_spaces/logs/*.csv"

files = glob.glob(path)

dfs = []

first_df = pd.read_csv(files[0])
dfs.append(first_df)

for f in files[1:]:
    df = pd.read_csv(f)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv("combined_search_spaces_1.csv", index=False)