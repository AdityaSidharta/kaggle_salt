import pandas as pd

def load_depth_file(depth_file):
    df_depth = pd.read_csv(depth_file)
    return df_depth