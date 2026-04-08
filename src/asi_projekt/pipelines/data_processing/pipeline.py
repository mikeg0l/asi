from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, preprocess, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline([
        node(
            func=preprocess,
            inputs=["raw_data", "parameters"],
            outputs="processed_data",
            name="preprocess_node",
        ),
        node(
            func=split_data,
            inputs=["processed_data", "parameters"],
            outputs=["X_train", "X_val", "X_test",
                     "y_train", "y_val", "y_test"],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train", "parameters"],
            outputs="trained_model",
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["trained_model", "X_val", "y_val"],
            outputs="metrics",
            name="evaluate_model_node",
        ),
    ])
