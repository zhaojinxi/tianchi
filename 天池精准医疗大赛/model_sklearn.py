import tensorflow
import sklearn.neural_network
import sklearn.svm
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.discriminant_analysis
import sklearn.kernel_ridge
import sklearn.neighbors
import sklearn.gaussian_process
import sklearn.cross_decomposition
import sklearn.naive_bayes
import sklearn.feature_selection
import sklearn.isotonic
import sklearn.calibration
import numpy

def build_model(which):
    if which == '线性模型':

        #1.广义线性模型
        ElasticNet = sklearn.linear_model.ElasticNet()
        return ElasticNet

    elif which == '判别分析':

        #2.Linear and Quadratic Discriminant Analysis
        pass

    elif which == '核岭回归':
        #3.Kernel ridge regression
        pass

    elif which == '支持向量机':
        #4.Support Vector Machines
        SVR = sklearn.svm.SVR()
        param_dist = {"C": [0.1, 1, 10], "epsilon": [0.01, 0.1, 1], "kernel": ['linear', 'rbf', 'sigmoid']}
        return SVR, param_dist

    elif which == '随机梯度下降':
        #5.Stochastic Gradient Descent
        pass

    elif which == '最邻近':
        #6.Nearest Neighbors
        KNeighborsRegressor = sklearn.neighbors.KNeighborsRegressor()
        return KNeighborsRegressor

    elif which == '高斯过程':
        #7.Gaussian Processes
        GaussianProcessRegressor = sklearn.gaussian_process.GaussianProcessRegressor()
        return GaussianProcessRegressor

    elif which == '交叉分解':
        #8.Cross decomposition
        PLSRegression = sklearn.cross_decomposition.PLSRegression()
        return PLSRegression

    elif which == '朴素贝叶斯':
        #9.Naive Bayes
        pass

    elif which == '决策树':
        #10.Decision Trees
        DecisionTreeRegressor = sklearn.tree.DecisionTreeRegressor()
        return DecisionTreeRegressor

    elif which == '集成方法':
        #11.Ensemble methods
        GradientBoostingRegressor = sklearn.ensemble.GradientBoostingRegressor()
        param_dist = {
            "loss": ['ls', 'lad', 'huber', 'quantile'], 
            "learning_rate": numpy.linspace(0.01,0.1,3), 
            "n_estimators": range(10, 100, 3), 
            "max_depth": [2, 4 ,6],
            'min_samples_leaf': [3, 5, 9, 11],
            'max_features': numpy.linspace(0.1,0.8,3)}
        return GradientBoostingRegressor, param_dist

    elif which == '多类与多标签':
        #12.多类与多标签算法
        pass

    elif which == '特征选择':
        #13.特征选择
        pass

    elif which == '半监督':
        #14.半监督
        pass

    elif which == '等渗回归':
        #15.等渗回归
        IsotonicRegression = sklearn.isotonic.IsotonicRegression()
        return IsotonicRegression

    elif which == '概率校准':
        #16.概率校准
        IsotonicRegression = sklearn.calibration.IsotonicRegression()
        return IsotonicRegression

    elif which == '神经网络':
        #17.Neural network models (supervised)
        MLPRegressor = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=[30,30,10], early_stopping = True)
        param_dist = { 
            "alpha":[0.00001, 0.0001, 0.001, 0.01], 
            'learning_rate':['constant', 'invscaling', 'adaptive'], 
            'learning_rate_init':[0.0001, 0.001, 0.01, 0.1], 
            }
        return MLPRegressor, param_dist

if __name__=='__main__':
    build_model()