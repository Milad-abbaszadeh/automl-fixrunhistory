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

for i in range(10):
    if i==0:
        print("first Run")
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=60,
            initial_configurations_via_metalearning = 0,
            ensemble_size=1,
            ensemble_nbest=1,
            write_history = True,
            read_history= False)
    else:
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=60,
            initial_configurations_via_metalearning = 0,
            ensemble_size=1,
            ensemble_nbest=1,
            write_history = False,
            read_history= True)


    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print('***************')
    print("RUN NUMBER {}".format(i))
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
    print("***************")
    with open("/home/dfki/Desktop/temp/pickel/file.txt", "a") as myFile:
        myFile.write("\n\n******** Result section {} ***********\n\n".format(i))
        myFile.write("******** ACCURACY ***********\n\n")
        myFile.write(str(sklearn.metrics.accuracy_score(y_test, y_hat)))
        myFile.write("\n\n-----------------------------\n\n")
        myFile.write(str(automl.show_models()))
        myFile.write("\n\n************************\n\n")
