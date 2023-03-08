import numpy as np
import xlrd
import matplotlib.pyplot as plt
import math
import pandas as pd
#数据输入
#data\ele con\
y=[]
x=[]
data0 = np.loadtxt(r'x1CO2.dat')
x=data0.tolist()
x1=x
#workbook2=xlrd.open_workbook(r'CO2HLC.xls')#打开excel
print(len(x))
print(len(x[2]))
data2 = pd.read_excel("CO2HLC.xls",header=None)
data2 = data2.values
HLC=data2[:,3]
y=HLC
print(len(HLC))

import numpy as np
import matplotlib.pyplot as plt
'''
list1=[14,19,20,21,25,27,28,30,33,35,38,39,40,41,42,47,49,50,52,53,54,56,58,59,61,63,65,67,76,77,78,79,81,87,90,91,93,97,99,100,101,108,109,110,113,114,117,118,120,121,125,127,129,133,134,135,136,138,145,153,154,158,160,161,163,166,167,168,169,171,172,173,174,176,179,182,195,197,199,200,205,206,208,209,210,213,214,216,224,227,228,230,234,237,238,240,241,244,246,248,249,252,253,258,266,267,268,272,274,281,294,296,300,303,306,310,311,319,322,323,324,325,327,328,330,333,334,335,336,337,339,340,343,347,348,349,351,362,363,364,372,373,379,380,381,382,383,384,386,391,392,401,425,433,437,438,444,445,450,453,454,457,459,460,465,468,469,475,479,484,486,495,497,502,506,508,509,511,514,515,516,517,519,551,562,566,573,574,575,576,578,582]
x1= [[each_list[i] for i in list1] for each_list in x1]
'''
########################################################################################


from pandas.core.frame import DataFrame
x1=DataFrame(x1)
#数据缩放
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#特征工程


print(len(x1))
print(len(x1[5]))
i=0
print(len(HLC))

########################################################################################
from feature_selector import  FeatureSelector
fs=FeatureSelector(data=x1,labels=y)
fs.identify_missing( missing_threshold=0.6)
fs.missing_stats.head()
fs.identify_single_unique()
fs.identify_collinear(correlation_threshold=0.79)

train_no_missing = fs.remove(methods = ['missing','single_unique','collinear'],keep_one_hot=False)
train_no_missing=np.array(train_no_missing)
x1=train_no_missing.tolist()
print(len(x1))
print(len(x1[5]))
########################################################################################
########################################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
x1=StandardScaler().fit_transform(x1)
print(len(x1))
print(len(x1[5]))
y=np.array(y)


#x1=DataFrame(x1)
#x1=DataFrame(x1)'''
########################################################################################
print(x1[0])
'''
x_train=x1[:200]
x_test=x1[200:]
y_train=y[:200]
y_test=y[200:]
'''
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=0)
#################################################
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
#######################################################################################################################
bestscpre=0
score1=[]
data=[]
for n_estimators in [1,5,8,10,15,20,30,40,50,60,70,80,90,100]:
  for max_depth in [1,5,8,10,15,20,25,30]:
     model=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
     model.fit(x_train,y_train)
     sco=mean_squared_error(y_test,model.predict(x_test))
     data.append([n_estimators,max_depth,sco])
     score=model.score(x_test,y_test)
     print(n_estimators,max_depth,sco)
     score1.append(score)
     if score>bestscpre:
      bestscpre=score
      bestparemeters={'n_estimators':n_estimators,'max_depth':max_depth}
print("模型最高分：{:.3f}".format(bestscpre))
print("最佳参数：{}".format(bestparemeters))

####################################################################
np.savetxt(r'C:\Users\Administrator\PycharmProjects\pythonProject\rf优化.dat',data,fmt='%.4f')