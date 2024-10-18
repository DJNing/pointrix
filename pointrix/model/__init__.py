from ..engine.default_datapipeline import BaseDataPipeline
from .base_model import BaseModel, MODEL_REGISTRY
from .point_cloud.gaussian_points import GaussianPointCloud
# from .point_cloud.points import PointCloud

__all__ = ["GaussianPointCloud", "BaseModel"]


def parse_model(cfg, datapipeline:BaseDataPipeline, device="cuda") -> BaseModel:
    """
    Parse the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    datapipeline : BaseDataPipeline
        The data pipeline.
    device : str
        The device to use.
    """
    name = cfg.pop("name")
    return MODEL_REGISTRY.get(name)(cfg, datapipeline, device)