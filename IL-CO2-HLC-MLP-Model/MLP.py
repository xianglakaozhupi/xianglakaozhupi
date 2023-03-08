import numpy as np
import xlrd
import matplotlib.pyplot as plt
import math
#数据输入
#data\ele con\
y=[]
x=[]
data0 = np.loadtxt(r'C:\Users\ASUS\Desktop\CO2HLC.dat')
x=data0.tolist()
workbook2=xlrd.open_workbook(r'C:\Users\ASUS\Desktop\CO2HLC.xls')#打开excel
print(len(x))
print(len(x[2]))
data2=workbook2.sheets()[0]
toxic=data2.col_values(3)
y=toxic
print(len(toxic))
########################################################################################
x1=x

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
print(len(toxic))
'''
########################################################################################
from feature_selector import  FeatureSelector
fs=FeatureSelector(data=x1,labels=y)
fs.identify_missing( missing_threshold=0.1)
fs.missing_stats.head()
fs.identify_single_unique()
fs.identify_collinear(correlation_threshold=0.79)

train_no_missing = fs.remove(methods = ['missing','single_unique','collinear'],keep_one_hot=True)
train_no_missing=np.array(train_no_missing)
x1=train_no_missing.tolist()
print(len(x1))
print(len(x1[5]))
########################################################################################
'''
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
# x_train=x1[:216]
# y_train=y[:216]
# x_test=x1[216:]
# y_test=y[216:]
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2)
#模型训练
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor

model=MLPRegressor(activation='tanh', alpha=0.2871763522095302, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(512, 10), learning_rate='invscaling',
       learning_rate_init=0.001, max_iter=100000000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=2,
       shuffle=True, solver='lbfgs', tol=1e-08, validation_fraction=0.1,
       verbose=False, warm_start=False)
model.fit(x_train,y_train)
#################################################################################################
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
'''
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x1,y,scoring='r2',cv=10)
print(scores)
print(np.mean(scores))
'''
from sklearn.model_selection import ShuffleSplit
#模型测试
# shufile=ShuffleSplit(test_size=0.2,train_size=0.8,n_splits=5)
# scores=cross_val_score(model,x1,y,scoring='r2',cv=shufile)
# print('5',scores)
# print('mean5',np.mean(scores))
print('训练R2得分',r2_score(y_train,model.predict(x_train)))
print('训练MAE得分',mean_absolute_error(y_train,model.predict(x_train)))
print('训练MSE得分',mean_squared_error(y_train,model.predict(x_train)))
print('测试R2得分',r2_score(y_test,model.predict(x_test)))
print('测试MAE得分',mean_absolute_error(y_test,model.predict(x_test)))
print('测试MSE得分',mean_squared_error(y_test,model.predict(x_test)))
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
ax.set_xlim(0,4)
ax.set_ylim(0,4)
ax.scatter(y_test,model.predict(x_test),marker="*",s=40,c=  '#FF0000')
ax.set_xlabel('Experimental value',font2)
ax.set_ylabel('Test set results',font2)
#x=np.linspace(0,10,100)
#ax.plot(x,x,'-b')
ax2=fig.add_subplot(122)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax2.set_xlim(0,4)
ax2.set_ylim(0,4)
ax2.scatter(y_train,model.predict(x_train),marker="*",s=40,c=  '#FF0000')
ax2.set_xlabel('Experimental value',font2)
ax2.set_ylabel('Train set results',font2)
#ax2.plot(x,x,'-b')
plt.savefig('jieguosuijisenlin.tif',figsize=[10,4])
plt.show()
###################################################################################################
###############################################################################################################################################################################