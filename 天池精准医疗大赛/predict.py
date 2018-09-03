import model_selection_and_evaluation
import model_sklearn
import data_loading
import data_transformations
import pandas
import model_tf

method='深度学习'

#读数据
[train_x, train_y, test_x]=data_loading.get_data()

#选择模型
model, param_dist=model_tf.build_model(method)

#调节超参数
best_model=model_selection_and_evaluation.Randomized_tuning_parameters(train_x, train_y, model, param_dist)

#训练
best_model.max_iter=10000
estimator=best_model.fit(train_x, train_y)

#模型评价
score=model_selection_and_evaluation.model_evaluation(estimator,train_x, train_y)
print(score)

#模型保存
model_selection_and_evaluation.model_persistence(method, estimator)

#预测结果
y_hat=estimator.predict(test_x).reshape(-1,1)
pandas.DataFrame(y_hat).to_csv('%s.csv' % method, index=False, header=False)