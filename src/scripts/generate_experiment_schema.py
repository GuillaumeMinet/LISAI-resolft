from pathlib import Path

from lisai.config.json_schema import (
    write_continue_training_json_schema,
    write_experiment_json_schema,
    write_experiment_template_json_schema,
    write_retrain_json_schema,
)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    outputs = [
        repo_root / "configs" / "schema" / "experiment.schema.json",
        repo_root / "configs" / "schema" / "experiment-template.schema.json",
        repo_root / "configs" / "schema" / "continue_training.schema.json",
        repo_root / "configs" / "schema" / "retrain.schema.json",
    ]
    writers = [
        write_experiment_json_schema,
        write_experiment_template_json_schema,
        write_continue_training_json_schema,
        write_retrain_json_schema,
    ]

    for writer, output_path in zip(writers, outputs, strict=True):
        writer(output_path)
        print(output_path)


if __name__ == "__main__":
    main()
