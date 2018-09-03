import pandas
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.feature_extraction

def preprocessing(train, test):

    train=train.dropna()
    temp_train_id=train.index
    test=test.dropna()
    temp_test_id=test.index
    name=train.columns

    #标准化
    ss=sklearn.preprocessing.StandardScaler()
    ss.fit(train)
    train_transform=ss.transform(train)
    test_transform=ss.transform(test)

    ##正规化
    #n=sklearn.preprocessing.Normalizer()
    #n.fit(train_transform)
    #train_transform=n.transform(train_transform)
    #test_transform=n.transform(test_transform)

    train_transform=pandas.DataFrame(train_transform, columns=name, index=temp_train_id)
    test_transform=pandas.DataFrame(test_transform, columns=name, index=temp_test_id)

    return train_transform, test_transform

def feature_selection():
    pass

def decomposition():
    pass