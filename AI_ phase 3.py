import pandas as pd
import seaborn as sns
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel("Assignment-1_Data.xlsx")
df.info()
df['Year']=df['Date'].apply(lambda x:x.split('.')[2])
df['Year']=df['Year'].apply(lambda x:x.split(' ')[0])
df['Month']=df['Date'].apply(lambda x:x.split('.')[1])
print(df)