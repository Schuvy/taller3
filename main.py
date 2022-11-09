import matplotlib.pyplot
import seaborn as sns
import matplotlib.pyplot as plt
import reliability as rl
import scipy.stats as ss
import pandas as pd
import pingouin as pg
import numpy as np
import openpyxl
import sklearn.linear_model as sk

reclaims = pd.read_excel('C:/Users/roman/Documents/Universidad/9no Semestre/Electiva 3/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx', sheet_name='Reclaims', header=6, usecols='A:C')

#Punto 1.A
crosstab = pd.crosstab(reclaims['Person'], reclaims['Reclaim'])
print(crosstab)
prueba_chi = ss.chi2_contingency(crosstab)
print('Valor Chi Cuadrado = ' + str(prueba_chi[0]) + ', p_value = ' + str(prueba_chi[1]) + ', grados de libertad = ' + str(prueba_chi[2]))
print('Frecuencias: ' + str(prueba_chi[3]))

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

#Punto 1.C
reclaims['Reclaim'] = reclaims['Reclaim'].replace(['Billing','Recalls','EB Contract','IBAN'], 'Casos A')
#no Hubo distinct, tamnb tocó manual
reclaims['Reclaim'] = reclaims['Reclaim'].replace(['Account closing','Status info','Matching','Stop payment'], 'Casos B')

crosstab = pd.crosstab(reclaims['Processing time'], reclaims['Reclaim'])
print(crosstab)

prueba_chi = ss.chi2_contingency(crosstab)
frecuencias = prueba_chi[3]
print('Valor Chi Cuadrado = ' + str(prueba_chi[0]) + ', p_value = ' + str(prueba_chi[1]) + ', grados de libertad = ' + str(prueba_chi[2]))
print('Frecuencias: ' + str(prueba_chi[3]))


x = crosstab['Casos A']
y = crosstab['Casos B']

reg = ss.linregress(x, y)

gsc = sns.scatterplot(x= x, y= y, label= "Raw data")
plt.plot(x, reg.intercept + reg.slope*x, 'r', label='fitted line')
plt.legend()
plt.show()

"""
correlacion = crosstab.corr(method='spearman')
print(correlacion)

corr = pg.pairwise_corr(crosstab, method='pearson')
corr.sort_values(by=['p-unc'])[['X', 'Y', 'n', 'r', 'p-unc']]
print(corr)
"""