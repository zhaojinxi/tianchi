import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

color = sns.color_palette()
sns.set_style('darkgrid')
import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

# 读取数据
train = pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
test = pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')
print('train shape',train.shape)
print('test shape',test.shape)
train_ID = train['id']
test_ID = test['id']
print('train feature shape',train.shape)
print('test feature shape',test.shape)

# 查看数据
print(train.head())
print(test.head())

# 查看特征列
print(train.columns)
data = pd.concat([train,test],axis=0)
print(data.isnull().sum()/len(data))

from pylab import mpl
from scipy.special import boxcox1p
lam = 0.15
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
t_f = train['甘油三酯']
fig ,ax = plt.subplots()
# tmp,lambda_  = stats.boxcox(train['血糖'])
ax.scatter(x = t_f,y=train['血糖'])
plt.ylabel('血糖')
plt.xlabel('甘油三酯')
plt.show()

# 血糖 is the variable we need to predict. So let's do some analysis on this variable first.
sns.distplot(train['血糖'],fit=norm)
(mu,sigma) = norm.fit(train['血糖'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('血糖分布')
fig = plt.figure()
res = stats.probplot(train['血糖'], plot=plt)
plt.show()

# 血糖 is the variable we need to predict. So let's do some analysis on this variable first.
from scipy.special import boxcox, inv_boxcox
tmp,lambda_  = stats.boxcox(train['血糖'])
# print(train['血糖'])
# print(inv_boxcox(tmp,lambda_))
sns.distplot(tmp,fit=norm)
(mu,sigma) = norm.fit(tmp)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.7f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('血糖分布')
fig = plt.figure()
res = stats.probplot(tmp, plot=plt)
plt.show()

# 特征工程
# 合并训练和测试数据
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['血糖'].values
all_data = pd.concat((train,test)).reset_index(drop=True)
all_data.drop(['血糖'],axis=1,inplace=True)
print("all_data size is : {}".format(all_data.shape))

# 查看缺失值比例
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data

corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.9,square=True)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

#Validation function
n_folds = 5
y_train,lambda_  = stats.boxcox(train['血糖'])
sex_map = {'男':1,'女':0,'??':0}
t_train = train.drop(['血糖'],axis=1)
t_train['性别'] = t_train['性别'].map(sex_map)
t_train = t_train.fillna(0)
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= -cross_val_score(model, t_train.values, y_train, scoring="neg_mean_squared_error", cv = kf)
    print(rmse)
    return(rmse)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

GBoost.fit(t_train.values,y_train)

x = GBoost.predict(t_train.values)
print(x)
print(sorted(list(inv_boxcox(x,lambda_))))

print(y_train)
print(inv_boxcox(y_train,lambda_))

from sklearn.metrics import mean_squared_error as mse
def mse_2(preds,train_data):
    labels = train_data.get_label()
    return 'mse_2',mse(y_true=labels ,y_pred=preds) / 2.0,False

print(mse(inv_boxcox(y_train,lambda_),inv_boxcox(x,lambda_))/2)

plt.plot(range(len(inv_boxcox(y_train,lambda_))),inv_boxcox(y_train,lambda_),'*')
plt.plot(range(len(inv_boxcox(x,lambda_))),inv_boxcox(x,lambda_),'-')
plt.show()

sub_test = test
print(sub_test)

del sub_test['id']
del sub_test['体检日期']
print(sub_test)
result = GBoost.predict(sub_test.values)
print(result)