import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from collections import Counter

df = pd.read_csv('./archive/Training Data.csv')
df = df.drop("Id", axis=1)
df.head()

from sklearn.model_selection import train_test_split
train, valid_test = train_test_split(df.copy(), test_size=0.2, random_state=42)
valid, test = train_test_split(valid_test.copy(), test_size=0.5, random_state=42)
