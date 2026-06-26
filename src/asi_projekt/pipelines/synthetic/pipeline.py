from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_synthetic_data, generate_synthetic_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_synthetic_data,
            inputs=["raw_data", "params:synthetic"],
            outputs="synthetic_data",
            name="generate_synthetic_node",
        ),
        node(
            func=evaluate_synthetic_data,
            inputs=["raw_data", "synthetic_data", "params:synthetic"],
            outputs="synthetic_scores",
            name="evaluate_synthetic_node",
        ),
    ])
