import seaborn as sns
import matplotlib as plt
import reliability as rl
import scipy.stats as ss
import pandas as pd
import numpy as np
import openpyxl

reclaims = pd.read_excel(
    'C:/Users/roman/Documents/Universidad/9no Semestre/Electiva 3/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx',
    sheet_name='Reclaims', header=6, usecols='A:B')

d1 = reclaims.groupby(['Reclaim','Person']).size()
print(d1)

e = 3

#:D 8)