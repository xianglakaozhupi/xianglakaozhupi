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
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#特征工程


print(len(x1))
print(len(x1[5]))
i=0
print(len(HLC))

########################################################################################
########################################################################################
from feature_selector import  FeatureSelector
fs=FeatureSelector(data=x1,labels=y)
fs.identify_missing( missing_threshold=0.1)
fs.missing_stats.head()
fs.identify_single_unique()
fs.identify_collinear(correlation_threshold=1)

train_no_missing = fs.remove(methods = ['missing','single_unique','collinear'],keep_one_hot=True)
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
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2)
#模型训练
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(max_depth=5,n_estimators=15)
model.fit(x_train,y_train)
#################################################################################################
#交叉验证
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x1,y,scoring='neg_mean_absolute_error',cv=10)
print(scores)
print(np.mean(scores))

print('训练R2得分',r2_score(y_train,model.predict(x_train)))
print('训练MAE得分',mean_absolute_error(y_train,model.predict(x_train)))
print('训练MSE得分',mean_squared_error(y_train,model.predict(x_train)))
print('测试R2得分',r2_score(y_test,model.predict(x_test)))
print('测试MAE得分',mean_absolute_error(y_test,model.predict(x_test)))
print('测试MSE得分',mean_squared_error(y_test,model.predict(x_test)))
xx=model.predict(x_test)
xxx=model.predict(x_train)
np.savetxt(r'C:\Users\Administrator\PycharmProjects\pythonProject\jieguo\x1CO2ceshi.dat',xx,fmt='%s')
np.savetxt(r'C:\Users\Administrator\PycharmProjects\pythonProject\jieguo\x1CO2cshiyan.dat',y_test,fmt='%s')
np.savetxt(r'C:\Users\Administrator\PycharmProjects\pythonProject\jieguo\x1CO2xulian.dat',xxx,fmt='%s')
np.savetxt(r'C:\Users\Administrator\PycharmProjects\pythonProject\jieguo\x1CO2xshiyan.dat',y_train,fmt='%s')
###################################################################################################

#绘图
from matplotlib.font_manager import FontProperties
font=FontProperties(fname=r'C:\Windows\Fonts\Times New Roman')

fig=plt.figure(figsize=(10,4), dpi=300)

ax=fig.add_subplot(121)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 12}
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.scatter(y_test,model.predict(x_test),marker="*",s=40,c=  '#FF0000')
ax.set_xlabel('Experimental value',font2)
ax.set_ylabel('Test set results',font2)
#x=np.linspace(0,10,100)
#ax.plot(x,x,'-b')
ax2=fig.add_subplot(122)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax2.scatter(y_train,model.predict(x_train),marker="*",s=40,c=  '#FF0000')
ax2.set_xlabel('Experimental value',font2)
ax2.set_ylabel('Train set results',font2)
#ax2.plot(x,x,'-b')
plt.savefig('jieguosuijisenlin.tif',figsize=[10,4])
plt.show()
###################################################################################################
###############################################################################################################################################################################