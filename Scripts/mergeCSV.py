"""

This script show you how to merge all CSV located in directory.

"""

import pandas as pd
import glob
interesting_files = glob.glob("*.csv")
df_list = []
for filename in sorted(interesting_files):
    df_list.append(pd.read_csv(filename))
full_df = pd.concat(df_list)

full_df.to_csv('merged.csv', index=False)