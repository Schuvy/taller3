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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
from reliability.Distributions import Lognormal_Distribution
from reliability.Probability_plotting import Lognormal_probability_plot

Caffeine2 = pd.read_excel('D:/Documentos/University/Electiva 3 Analítica de datos para procesos industriales/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx',
                         sheet_name = 'Caffeine-2', index_col = 0, header = 7, usecols = 'A:C')

Caffeine3 = pd.read_excel('D:/Documentos/University/Electiva 3 Analítica de datos para procesos industriales/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx',
                         sheet_name = 'Caffeine-3', header = 7, usecols = 'A:C')

departments = pd.read_excel('D:/Documentos/University/Electiva 3 Analítica de datos para procesos industriales/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx',
                         sheet_name = 'Departments', header = 6, usecols = 'A:B')

teaBags = pd.read_excel('D:/Documentos/University/Electiva 3 Analítica de datos para procesos industriales/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx',
                         sheet_name = 'Tea bags', header = 7, usecols = 'A:C')

Picking = pd.read_excel('D:/Documentos/University/Electiva 3 Analítica de datos para procesos industriales/_fb205379f16688a0d5fff594e91ac7e7_data_files_DA-LSS.xlsx',
                         sheet_name = 'Picking', header = 7, usecols = 'A:C')

def boxhist(x, col):
    f, (ax_box, ax_hist, ax_conf) = plt.subplots(3, sharex=True, gridspec_kw={"height_ratios": (.15, .60, .25)},
                                                 figsize=(8, 6))
    h = sns.histplot(data=x, x=col, bins=12, kde=True, ax=ax_hist)
    bp = sns.boxplot(data=x, x=col, ax=ax_box)

    FDS = x[col].describe()
    FDS.loc["Variance"] = [FDS.loc["std"] ** 2]
    Scipy_desc = ss.describe(x[col])
    FDS.loc["Skewness"] = Scipy_desc.skewness
    FDS.loc["Kurtosis"] = Scipy_desc.kurtosis

    forMean = ss.t.interval(alpha=0.95, df=len(x) - 1, loc=FDS.loc["mean"], scale=ss.sem(x[col]))
    forMedian = ss.t.interval(alpha=0.95, df=len(x) - 1, loc=FDS.loc["50%"], scale=ss.sem(x[col]))
    forStd = ss.t.interval(alpha=0.95, df=len(x) - 1, loc=FDS.loc["std"], scale=ss.sem(x[col]))

    FDS.loc["Conf. Intr. 95% for Mean"] = [(forMean[0], forMean[1])]
    FDS.loc["Conf. Intr. 95% for Median"] = [(forMedian[0], forMedian[1])]
    FDS.loc["Conf. Intr. 95% for Std"] = [(forStd[0], forMedian[1])]

    xmean = np.linspace(FDS.loc["Conf. Intr. 95% for Mean"][0][0], FDS.loc["Conf. Intr. 95% for Mean"][0][1], 100)
    xmedian = np.linspace(FDS.loc["Conf. Intr. 95% for Median"][0][0], FDS.loc["Conf. Intr. 95% for Median"][0][1], 100)

    print(FDS.loc["mean"])

    id = sns.lineplot(x=xmean, y=0.1, ax=ax_conf, color="b")
    id = sns.lineplot(x=xmedian, y=0.2, ax=ax_conf, color="b")
    id = sns.scatterplot(x=FDS.loc["Conf. Intr. 95% for Mean"][0], y=(0.1, 0.1), color="b", marker="|", s=80)
    id = sns.scatterplot(x=FDS.loc["mean"], y=[0.1], color="b")
    id = sns.scatterplot(x=FDS.loc["Conf. Intr. 95% for Median"][0], y=(0.2, 0.2), color="b", marker="|", s=80)
    id = sns.scatterplot(x=FDS.loc["50%"], y=[0.2], color="b")
    id.set_yticks(np.array([0.1, 0.2]))
    id.set(ylim=(0.05, 0.25))
    id.set_yticklabels(['Mean', 'Median'])
    id.grid()

    plt.show()
    return FDS



#Punto 2 a

#Tomamos el gráfico 3
boxhist(Caffeine2, 'Extraction time') #Continua

#Tomamos el gráfico 2
boxhist(Caffeine3, 'Extractor nr.') #Discreta

#Tomamos la 1
boxhist(departments, 'Throughput times') # Continua

#Tomamos la 2
#boxhist(teaBags, 'Stops') #Discreta

#Picking 1
boxhist(Picking, 'Items') #Discreta


#Punto 2 b
#Caffeine-2 Caffeine Content
Lognormal_probability_plot(failures = Caffeine2["Caffeine Content"].to_list()) #generates the probability plot
plt.legend()
plt.show()


#Caffeine3, Extractor nr.
Lognormal_probability_plot(failures = Caffeine3["Extractor nr."].to_list()) #generates the probability plot
plt.legend()
plt.show()

#departments, Throughput times
Lognormal_probability_plot(failures = departments["Throughput times"].to_list()) #generates the probability plot
plt.legend()
plt.show()

#Picking, Items
#paramsN = ss.norm.fit(Picking["Items"])
Lognormal_probability_plot(failures = Picking["Items"].to_list()) #generates the probability plot
plt.legend()
plt.show()



#Punto 2 c
#Caffeine-2 Caffeine Content
paramsN = ss.norm.fit(Caffeine2["Caffeine Content"])
paramsL = ss.lognorm.fit(Caffeine2["Caffeine Content"], floc = 0)
dist = Lognormal_Distribution(mu = np.log(paramsL[2]), sigma = paramsL[0])

dist.CDF(linestyle='--',label='True CDF') #this is the actual distribution provided for comparison
plt.legend()
plt.show()

#Caffeine3, Extractor nr.
paramsN = ss.norm.fit(Caffeine3["Extractor nr."])
paramsL = ss.lognorm.fit(Caffeine3["Extractor nr."], floc = 0)
dist = Lognormal_Distribution(mu = np.log(paramsL[2]), sigma = paramsL[0])

dist.CDF(linestyle='--',label='True CDF') #this is the actual distribution provided for comparison
plt.legend()
plt.show()

#departments, Throughput times
paramsN = ss.norm.fit(departments["Throughput times"])
paramsL = ss.lognorm.fit(departments["Throughput times"], floc = 0)
dist = Lognormal_Distribution(mu = np.log(paramsL[2]), sigma = paramsL[0])

dist.CDF(linestyle='--',label='True CDF') #this is the actual distribution provided for comparison
plt.legend()
plt.show()

#Picking, Items
paramsN = ss.norm.fit(Picking["Items"])
paramsL = ss.lognorm.fit(Picking["Items"], floc = 0)
dist = Lognormal_Distribution(mu = np.log(paramsL[2]), sigma = paramsL[0])

dist.CDF(linestyle='--',label='True CDF') #this is the actual distribution provided for comparison
plt.legend()
plt.show()



#Punto 2 d