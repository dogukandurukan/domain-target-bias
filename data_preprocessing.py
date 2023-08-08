import pandas as pd
import json
import os
import json
from collections import Counter
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
from openpyxl import load_workbook


data_pickle = "/Users/dogukandurukan/Desktop/Thesis/domain-target-bias/datasets/intrasentence/df_intrasentence_en.pkl"

data = pd.read_pickle(data_pickle)

with open("/Users/dogukandurukan/Desktop/Thesis/stereotypes-multi/results/predictions/bert-base-multilingual-cased_BertLM_BertNextSentence_en.json", "r") as f:
    json_df = json.load(f)

#type(json_df), json_df[0] if isinstance(json_df, list) else list(json_df.keys())

intrasentence_df = pd.DataFrame(json_df['intrasentence'])

# Merge on c1_id and create a new 'intersentence' column
c1_merged = data.merge(intrasentence_df, left_on='c1_id', right_on='id', how='left',copy= False)
c1_merged = c1_merged.drop(columns=['id_y']).rename(columns={"score": "c1_intrasentence_score", 'id_x': 'id'})

# Merge on c2_id and create a new 'intersentence' column
c2_merged = data.merge(c1_merged, left_on='c2_id', right_on='id', how='left').rename(columns={"score": "c2_intrasentence_score"})

# Merge on c3_id and create a new 'intersentence' column
c3_merged = data.merge(c2_merged, left_on='c3_id', right_on='id', how='left').rename(columns={"score": "c3_intrasentence_score"})

# Clean up by dropping the additional 'id' columns that might have been created during merging
data.drop(columns=[col for col in data.columns if col.startswith("id_")], inplace=True)


pd.set_option('display.max_columns', None)  # Display all columns
#pd.set_option('display.max_rows', None)     # Display all rows

# Print the dataframe
print(c3_merged.head())

