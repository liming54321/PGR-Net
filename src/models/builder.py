from copy import deepcopy

from easytorch.utils.registry import Registry

MODEL_REGISTRY = Registry('Model')


def build_model(model_name, params):
    params = deepcopy(params)
    model = MODEL_REGISTRY.get(model_name)(**params)
    # logger = get_root_logger()
    # logger.info(f'Network [{net.__class__.__name__}] is created.')
    return model
