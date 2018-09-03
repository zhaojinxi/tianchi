import sklearn.model_selection
import sklearn.metrics
import sklearn.externals

def cross_validation(model, x, y):
    #1.交叉验证:评价估计器性能
    print(sklearn.model_selection.cross_val_score(model, x, y, cv=5, scoring='mean_squared_error'))

def Randomized_tuning_parameters(x, y, model, param_dist):
    #2.调节估计器的超参数
    RandomizedSearchCV=sklearn.model_selection.RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, scoring='neg_mean_squared_error')
    RandomizedSearchCV.fit(x, y)
    return RandomizedSearchCV.best_estimator_

def Grid_tuning_parameters(x, y, model, param_dist):
    #2.调节估计器的超参数
    GridSearchCV=sklearn.model_selection.GridSearchCV(model, param_grid=param_dist, cv=3, scoring='neg_mean_squared_error')
    GridSearchCV.fit(x, y)
    return GridSearchCV.best_estimator_

def model_evaluation(model, x, y):
    #3.模型评价:量化预测的质量
    y1=model.predict(x)
    score=sklearn.metrics.mean_squared_error(y1, y)
    return score

def model_persistence(method, model):
    #4.模型持久化
    sklearn.externals.joblib.dump(model, '%s.model' % method)

def validation_curves():
    #5.验证曲线:绘制分数来评价模型
    sklearn.model_selection.validation_curve(model, train, label)

    sklearn.model_selection.learning_curve(model, train, label)