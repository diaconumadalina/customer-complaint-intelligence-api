from functools import lru_cache
from model.inference import ComplaintInferencePipeline


@lru_cache(maxsize=1)
def get_pipeline():
    return ComplaintInferencePipeline()
