import openml
import autosklearn.classification
import pickle
import sklearn.model_selection



#load the dataset credit g
task = openml.tasks.get_task(31)
pickle.dump(task, open( "task.p", "wb" ))
task = pickle.load(open( "task.p", "rb" ))

X, y = task.get_X_and_y()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=180,
    initial_configurations_via_metalearning = 0,
    tmp_folder ='/home/dfki/Desktop/temp/temp',
    output_folder = '/home/dfki/Desktop/temp/out',
    # metadata_directory='/home/dfki/Desktop/Thesis/AutoML/auto-sklearn/autosklearn/metalearning',
    delete_tmp_folder_after_terminate = False,
    delete_output_folder_after_terminate = False,
    ensemble_size=1,
    ensemble_nbest=1,
    write_history = False,
    read_history= True
)




automl.fit(X_train, y_train)

y_hat = automl.predict(X_test)
print('@@@@@@@@@@@@')
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
print("***************")
print(automl.show_models())
# print("****************")
# print(automl.sprint_statistics())
# print("*******************")
print(automl.get_models_with_weights)
print("*******************")
print(automl.sprint_statistics())
# print(automl.cv_results_)
# print(automl.trajectory_)
# print(automl.get_configuration_space(X_train, y_train))

# from autosklearn.pipeline.components.base import AutoSklearnChoice
# x= AutoSklearnChoice(dataset_properties=None)
# x.get_hyperparameter_search_space(dataset_properties={
#   'task': 1,
#   'sparse': False,
#   'multilabel': False,
#   'multiclass': False,
#   'target_type': 'classification',
#   'signed': False}).get_default_configuration()
