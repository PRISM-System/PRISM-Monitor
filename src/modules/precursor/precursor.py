import os

from src.test_scenarios.modeling import TestScenarioModel
from src.modules.util.util import dataframe_to_json_serializable


def event_precursor(model: TestScenarioModel, target_process, start, end, serialize: bool = False):

    result = model.forecasting_predict(target_process, start, end)
    if serialize:
        return dataframe_to_json_serializable(result)
    return result

