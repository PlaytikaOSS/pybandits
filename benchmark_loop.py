import time
import numpy as np
import pybandits.model
from pybandits.quantitative_model import CmabZoomingModel, BaseCmabZoomingModel, BaseSmabZoomingModel, SmabZoomingModel
from unittest.mock import MagicMock

def benchmark_cmab_inner_update():
    n_samples = 10000
    n_features = 10
    dimension = 1
    model = CmabZoomingModel.cold_start(dimension=dimension, base_model_cold_start_kwargs={"n_features": n_features})

    # Mock the sub-actions update method to focus on loop overhead
    for sub_action in model.sub_actions.values():
        sub_action.update = MagicMock()

    # Generate some data
    quantities = np.random.rand(n_samples).tolist()
    rewards = np.random.randint(0, 2, n_samples).tolist()
    context = np.random.rand(n_samples, n_features)

    # Map values to segments
    segments = model._map_values_to_segments(quantities)

    start_time = time.time()
    model._inner_update(segments, rewards, context)
    end_time = time.time()

    print(f"CMAB _inner_update overhead (n_samples={n_samples}, n_segments={len(model.sub_actions)}): {end_time - start_time:.4f} seconds")

def benchmark_smab_inner_update():
    n_samples = 10000
    dimension = 1
    model = SmabZoomingModel.cold_start(dimension=dimension)

    # Mock the sub-actions update method
    for sub_action in model.sub_actions.values():
        sub_action.update = MagicMock()

    quantities = np.random.rand(n_samples).tolist()
    rewards = np.random.randint(0, 2, n_samples).tolist()

    segments = model._map_values_to_segments(quantities)

    start_time = time.time()
    model._inner_update(segments, rewards)
    end_time = time.time()

    print(f"SMAB _inner_update overhead (n_samples={n_samples}, n_segments={len(model.sub_actions)}): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_cmab_inner_update()
    benchmark_smab_inner_update()
