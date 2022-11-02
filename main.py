import matplotlib.pyplot
import seaborn as sns
import matplotlib.pyplot as plt
import reliability as rl
import scipy.stats as ss
import pandas as pd
import numpy as np
import openpyxl

reclaims = pd.read_excel('C:/Users/roman/Documents/Universidad/9no Semestre/Electiva 3/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx', sheet_name='Reclaims', header=6, usecols='A:C')

"""
pendiente, intercepto, r_value, p_value, std_err = ss.linregress(reclaims['Person'], reclaims['Reclaim'])
gsc = sns.regplot(data = reclaims, x = 'Person', y = 'Reclaim', scatter = True, color = 'blue',
                 line_kws={'label':'y={0:.1f}x+{1:.1f}, r_value = {2:.1f}, p_value = {3:.1f}'.format(pendiente, intercepto, r_value, p_value)})
plt.legend(loc = 'upper right')
plt.show()
"""

datos = reclaims.groupby(by=['Person'])['Reclaim'].value_counts()
datos.to_dict()

#Punto 1.A
print(ss.chi2_contingency(datos))

#Punto 1.B
personas = reclaims.groupby(['Person'])['Processing time'].apply(list).to_dict()

for persona in personas:
    sns.distplot(personas[persona])
    sts, p_value = ss.shapiro(personas[persona])
    if p_value > 0.05:
        print('El P_value arrojado por el proceso shapiro Wilks arrojó un pvalue de: ' + str(p_value) + ' por lo que los datos examinados de ' + persona + ', si provienen de una distribución normal')
    else:
        print(
            'El P_value arrojado por el proceso shapiro Wilks arrojó un pvalue de: ' + str(p_value) + ' por lo que los datos examinados de ' + persona + ', no provienen de una distribución normal')
    plt.xlabel(persona)
    plt.ylabel('Timepos')
    plt.show()