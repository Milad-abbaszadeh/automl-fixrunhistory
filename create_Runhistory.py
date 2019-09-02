from smac.configspace import Configuration
from autosklearn.smbo import AutoMLSMBO
from autosklearn.automl import AutoML
from autosklearn.automl import BaseEstimator
from autosklearn.data.abstract_data_manager import AbstractDataManager
import pickle
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger


def runhistory_builder(ta,scenario_dic,rng):

    tae_runner  = ExecuteTARun(ta=ta)
    scenario = Scenario(scenario_dic)
    stats = Stats(scenario=scenario)
    traj_logger = TrajLogger(stats=stats,output_dir="/home/dfki/Desktop/temp")

    # if tae_runner.stats is None:
    #     new_smac =SMAC(scenario=scenario,tae_runner=tae_runner)
    #     tae_runner.stats = new_smac.stats

    stats.start_timing()
    deful_config_builder = DefaultConfiguration(tae_runner,scenario,stats,traj_logger,rng)
    config_milad =deful_config_builder._select_configuration()
    config_milad._values = None
    config_milad._values = {'balancing:strategy': 'none', 'categorical_encoding:__choice__': 'one_hot_encoding', 'classifier:__choice__': 'random_forest', 'imputation:strategy': 'mean', 'preprocessor:__choice__': 'no_preprocessing', 'rescaling:__choice__': 'standardize', 'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'True', 'classifier:random_forest:bootstrap': 'True', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:max_depth': 10, 'classifier:random_forest:max_features': 0.5, 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:min_impurity_decrease': 0.0, 'classifier:random_forest:min_samples_leaf': 1, 'classifier:random_forest:min_samples_split': 2, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'classifier:random_forest:n_estimators': 100, 'categorical_encoding:one_hot_encoding:minimum_fraction': 0.01}
    # config_milad._values = {'balancing:strategy': 'none',
    #  'categorical_encoding:__choice__': 'no_encoding',
    #  'classifier:__choice__': 'random_forest',
    #  'imputation:strategy': 'mean',
    #  'preprocessor:__choice__': 'pca',
    #  'preprocessor:copy':True,
    #  'preprocessor:iterated_power':'auto',
    #  'preprocessor:n_components':'None',
    #  'preprocessor:random_state':'None',
    #  'preprocessor:svd_solver':'auto',
    #  'preprocessor:tol':0.0,
    #  'preprocessor:whiten':'False',
    #  'rescaling:__choice__': 'None',
    #  'classifier:random_forest:bootstrap': 'True',
    #  'classifier:random_forest:class_weight': 'None',
    #  'classifier:random_forest:criterion': 'gini',
    #  'classifier:random_forest:max_depth': 'None',
    #  'classifier:random_forest:max_features': 'auto',
    #  'classifier:random_forest:max_leaf_nodes': 'None',
    #  'classifier:random_forest:min_impurity_decrease': 0.0,
    #  'classifier:random_forest:min_impurity_split': '1e-07',
    #  'classifier:random_forest:min_samples_leaf': 1,
    #  'classifier:random_forest:min_samples_split': 2,
    #  'classifier:random_forest:min_weight_fraction_leaf': 0.0,
    #  'classifier:random_forest:n_estimators': 10,
    #  'classifier:random_forest:n_jobs': 1,
    #  'classifier:random_forest:oob_score': 'False',
    #  'classifier:random_forest:random_state': 'None',
    #  'classifier:random_forest:verbose': 0,
    #  'classifier:random_forest:warm_start': 'False',
    # }
    # config_milad._vector =None


    status, cost, runtime, additional_info = tae_runner.start(config=config_milad,instance=None)



    print(status, cost, runtime, additional_info)

    runhistory = RunHistory(aggregate_func=average_cost)
    runhistory.add( config=config_milad,
                    cost=cost,
                    time=runtime,
                    status=status,
                    instance_id=None,
                    additional_info=additional_info)

    return runhistory



