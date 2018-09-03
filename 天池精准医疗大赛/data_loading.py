import pandas
import data_transformations
import numpy
import datetime
import time

def write_pkl():
    train=pandas.read_csv('data/d_train_20180102.csv', encoding='gbk')
    test=pandas.read_csv('data/d_test_A_20180102.csv', encoding='gbk')
    train.index=train['id']
    test.index=test['id']

    #拆分 y_
    y_=train[['血糖']]

    #观察可知数据有5个检测项目，只要检测了某个项目，该项目数据就不会缺失，每个人不一定检测了全部项目，先把数据分为5组
    train_蛋白=train[['*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例']]
    test_蛋白=test[['*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例']]

    train_胆固醇=train[['甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']]
    test_胆固醇=test[['甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇']]

    train_尿=train[['尿素', '肌酐', '尿酸']]
    test_尿=test[['尿素', '肌酐', '尿酸']]

    train_乙肝=train[['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']]
    test_乙肝=test[['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']]

    train_血=train[['白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积', '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%']]
    test_血=test[['白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度', '血小板计数', '血小板平均体积', '血小板体积分布宽度', '血小板比积', '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%']]

    #缺失蛋白1221， 缺失胆固醇1219，缺失尿1378，缺失乙肝4279，缺失血23
    #蛋白和血至少检测了其中一项，胆固醇和血至少检测了其中一项，尿和血至少检测了其中一项
    o=train.index.tolist()
    a=train_蛋白.dropna(how='any').index.tolist()
    b=train_胆固醇.dropna(how='any').index.tolist()
    c=train_尿.dropna(how='any').index.tolist()
    d=train_乙肝.dropna(how='any').index.tolist()
    e=train_血.dropna(how='any').index.tolist()
    train_缺蛋白=set(o)-set(a)
    train_缺胆固醇=set(o)-set(b)
    train_缺尿=set(o)-set(c)
    train_缺乙肝=set(o)-set(d)
    train_缺血=set(o)-set(e)

    #缺失蛋白185， 缺失胆固醇176，缺失尿194，缺失乙肝831，缺失血6
    #蛋白和血至少检测了其中一项，胆固醇和血至少检测了其中一项，尿和血至少检测了其中一项
    o=test.index.tolist()
    a=test_蛋白.dropna(how='any').index.tolist()
    b=test_胆固醇.dropna(how='any').index.tolist()
    c=test_尿.dropna(how='any').index.tolist()
    d=test_乙肝.dropna(how='any').index.tolist()
    e=test_血.dropna(how='any').index.tolist()
    test_缺蛋白=set(o)-set(a)
    test_缺胆固醇=set(o)-set(b)
    test_缺尿=set(o)-set(c)
    test_缺乙肝=set(o)-set(d)
    test_缺血=set(o)-set(e)

    #拆分 时间
    train_日期=train['体检日期']
    test_日期=test['体检日期']
    start_time=pandas.to_datetime(pandas.concat([train_日期, test_日期]).min())
    train_日期=(pandas.to_datetime(train_日期) - start_time).dt.days
    test_日期=(pandas.to_datetime(test_日期) - start_time).dt.days

    #拆分 年龄
    train_年龄=train['年龄']
    test_年龄=test['年龄']

    #拆分 离散数据
    train_性别=train['性别']
    train_性别=train_性别.map({'男':1, '女':2})
    test_性别=test['性别']
    test_性别=test_性别.map({'男':1, '女':2})
    train_性别=train_性别.fillna(pandas.concat([train_性别, test_性别]).median())
    test_性别=test_性别.fillna(pandas.concat([train_性别, test_性别]).median())   

    #数据预处理
    train_蛋白, test_蛋白=data_transformations.preprocessing(train_蛋白, test_蛋白)
    train_胆固醇, test_胆固醇=data_transformations.preprocessing(train_胆固醇, test_胆固醇)
    train_尿, test_尿=data_transformations.preprocessing(train_尿, test_尿)
    train_乙肝, test_乙肝=data_transformations.preprocessing(train_乙肝, test_乙肝)
    train_血, test_血=data_transformations.preprocessing(train_血, test_血)

    #合并 所需字段
    x=pandas.concat([train_蛋白, train_胆固醇, train_尿, train_乙肝, train_血, train_性别, train_年龄, train_日期], axis=1)
    x_pre=pandas.concat([test_蛋白, test_胆固醇, test_尿, test_乙肝, test_血, test_性别, test_年龄, test_日期], axis=1)

    #填补 有空的项
    x=x.fillna(0)
    x_pre=x_pre.fillna(0)

    x=x.astype(numpy.float32)
    y_=y_.astype(numpy.float32)
    x_pre=x_pre.astype(numpy.float32)

    x.to_pickle('x.pkl')
    y_.to_pickle('y_.pkl')
    x_pre.to_pickle('x_pre.pkl')

def get_data_1d():
    train_x=pandas.read_pickle('x.pkl')
    train_y=pandas.read_pickle('y_.pkl')
    test_x=pandas.read_pickle('x_pre.pkl')

    train_x=train_x.as_matrix()
    train_y=train_y.as_matrix().reshape(-1,1)
    test_x=test_x.as_matrix()

    train_x=train_x.reshape(-1,40,1)
    test_x=test_x.reshape(-1,40,1)

    train_x=train_x.astype(numpy.float32)
    train_x=train_x.astype(numpy.float32)

    return train_x, train_y, test_x

def get_data_2d():
    train_x=pandas.read_pickle('x.pkl')
    train_y=pandas.read_pickle('y_.pkl')
    test_x=pandas.read_pickle('x_pre.pkl')

    train_x=train_x.as_matrix()
    train_y=train_y.as_matrix().reshape(-1,1)
    test_x=test_x.as_matrix()

    train_x=numpy.concatenate((train_x, numpy.zeros([train_x.shape[0],9])), axis=1).reshape(-1,7,7,1)
    test_x=numpy.concatenate((test_x, numpy.zeros([test_x.shape[0],9])), axis=1).reshape(-1,7,7,1)

    train_x=train_x.astype(numpy.float32)
    test_x=test_x.astype(numpy.float32)

    return train_x, train_y, test_x

def get_data_3d():
    train_x=pandas.read_pickle('x.pkl')
    train_y=pandas.read_pickle('y_.pkl')
    test_x=pandas.read_pickle('x_pre.pkl')

    train_x=train_x.as_matrix()
    train_y=train_y.as_matrix().reshape(-1,1)
    test_x=test_x.as_matrix()

    train_x=numpy.concatenate((train_x, numpy.zeros([train_x.shape[0],24])), axis=1).reshape(-1,4,4,4,1)
    test_x=numpy.concatenate((test_x, numpy.zeros([test_x.shape[0],24])), axis=1).reshape(-1,4,4,4,1)

    train_x=train_x.astype(numpy.float32)
    test_x=test_x.astype(numpy.float32)

    return train_x, train_y, test_x


if __name__=='__main__':
    get_data_3dimension()