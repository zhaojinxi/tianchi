import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
chinese_font=FontProperties(fname='usr/share/fonts/MyFonts/YaHei.Consolas.1.11b.ttf')
print('\n Processing data for LightGBM')
df_train=pd.read_csv('./data/DM/d_train_20180102.csv', encoding='gb18030')
print(df_train.shape)

missing_df=df_train.isnull().sum(axis=0).reset_index()
missing_df.columns=['column_name','missing_count']
missing_df=missing_df.loc[missing_df['missing_count']>0]
missing_df=missing_df.sort_values(by='missing_count')

ind=np.arange(missing_df.shape[0])
width=0.9
fig,ax=plt.subplots(figsize=(12,18))
rects=ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values,rotation='horizontal',fontproperties=chinese_font)
ax.set_xlabel('Count of missing values')
ax.set_title('Number of missing values in each column')
plt.show()

print(df_train.columns)

print('\n Processing train')
pring(df_train.isnull().sum()/len(df_train))

print('\n Processing test')
pring(df_test.isnull()/sum()/len(df_test))

plt.figure(figsize=(8,6))
plt.scatter(range(df_train.shape[0]),np.sort(df_train['血糖'].values))
plt.xlabel('index',fontsize=12)
plt.ylabel('longerror',fontsize=12)
plt.show()

ulimit=np.percentile(df_train['血糖'].values, 99)
llimit=np.percentile(df_train['血糖'].values, 1)
df_train['血糖'].ix[df_train['血糖']>ulimit] = ulimit
df_train['血糖'].ix[df_train['血糖']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(df_train['血糖'].values, bins=50, kde=False)
plt.xlabel('血糖', fontsize=12)
plt.show()

ulimit = np.percentile(df_train['年龄'].values, 99)
llimit = np.percentile(df_train['年龄'].values, 1)
df_train['年龄'].ix[df_train['年龄']>ulimit] = ulimit
df_train['年龄'].ix[df_train['年龄']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(df_train['年龄'].values, bins=50, kde=False)
plt.xlabel('年龄', fontsize=12)
plt.show()

dtype_df = df_train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df

mean_values = df_train.mean(axis=0)
df_train_new = df_train.fillna(mean_values, inplace=True)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in df_train_new.columns if col not in ['血糖'] if df_train_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(df_train_new[col].values, df_train_new['血糖'].values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal',fontproperties=chinese_font)
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")

cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = df_train[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))
ax.set_xticklabels(corr_df_sel.col_labels.values, rotation='horizontal',fontproperties=chinese_font)
ax.set_yticklabels(corr_df_sel.col_labels.values, rotation='horizontal',fontproperties=chinese_font)
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15,fontproperties=chinese_font)
plt.show()

col = "年龄"
ulimit = np.percentile(df_train[col].values, 99.5)
llimit = np.percentile(df_train[col].values, 0.5)
df_train[col].ix[df_train[col]>ulimit] = ulimit
df_train[col].ix[df_train[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=df_train['年龄'].values, y=df_train['血糖'].values, size=10)
plt.ylabel('血糖', fontsize=12,fontproperties=chinese_font)
plt.xlabel('年龄', fontsize=12,fontproperties=chinese_font)
plt.title("年龄 Vs 血糖", fontsize=15,fontproperties=chinese_font)
plt.show()

ggplot(aes(x='年龄', y='血糖'), data=df_train) + geom_point(color='steelblue', size=1) + stat_smooth()

train_y = df_train['血糖'].values

df_train=df_train.drop(['id', '性别', '体检日期','血糖'], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(df_train, train_y, feature_names=df_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
ax.set_yticklabels(df_train.columns.values, rotation='horizontal',fontproperties=chinese_font)
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
