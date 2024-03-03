import pandas as pd

data = pd.read_csv("ind30_m_ew_rets.csv")
data = data.set_index("Unnamed: 0")
data.index = pd.to_datetime(data.index, format = "%Y%m")
data.columns = data.columns.str.strip()
data = data / 100
data = data["2000"]
