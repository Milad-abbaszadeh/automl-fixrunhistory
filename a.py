import openml
from sklearn import neighbors
import os
import pandas as pd
from pprint import pprint
from openml import runs, flows, setups, evaluations

class Flow(object):
    def __init__(self):
        pass

    def flow_to_sklearn_with_hack(self,flow):
        copyFlow = flow
        if copyFlow.flow_id == 5804:
            copyFlow.dependencies = u'sklearn==0.20.3\nnumpy>=1.13.0\nscipy>=1.0'
            for v in copyFlow.components.values():
                v.dependencies = u'sklearn==0.20.3\nnumpy>=1.13.0\nscipy>=1.0'

            return flows.flow_to_sklearn(copyFlow)

        else:
            return flows.flow_to_sklearn(copyFlow)


print(openml.__version__)

# load the flow
# flow = openml.flows.get_flow(5804)

# run= openml.runs.get_run(2083190)
# print(run)

# print(run.parameter_settings)
# print(run.setup_id)
# print(run.setup_string)
# print(run.)

###########################################
# import pandas
# rl = runs.list_runs(task=[31],size=100)
# pandas_runs= pandas.DataFrame.from_dict(rl, orient='index')[1:5]
# setup_ids=pandas_runs['setup_id']
# print(list(setup_ids))
#############################################

#
# setup = openml.setups.get_setup(216724)
#
# config_model=['Configuration:']
# classifier = 'random_forest'
# for i,k in setup.parameters.items():
#     print(k.full_name,k.value)
#     # if (k.parameter_name =='steps'):
#     #     for i in k.value:
#     #         print(i)
#     # print(k.parameter_name,k.value)
#     # config_model.append("classifier:{}:{}, Value: {}".format(classifier,k.parameter_name,k.value))
#



# print(config_model)
#############################################




# setup50= setup.parameters[50740]
# print(
# setup50.id,
# setup50.flow_id,
# setup50.flow_name,
# setup50.full_name,
# setup50.parameter_name,
# setup50.data_type ,
# setup50.default_value,
# setup50.value)

# print(setup50.flow_id)
# print(setup50.full_name)
# print(setup50.parameter_name)
# print(setup50.value)



        # self.parameter_name = parameter_name
        # self.data_type = data_type
        # self.default_value = default_value
        # self.value = value
# run = openml.runs.initialize_model_from_run(2083190)
# print(run)
# flow_changer = Flow()
# pipeline1 = flow_changer.flow_to_sklearn_with_hack(flow) #version problem
#
# print(pipeline1)

flow = openml.flows.get_flow(5804)
flow_changer = Flow()
pipeline1 = flow_changer.flow_to_sklearn_with_hack(flow)
print(pipeline1)