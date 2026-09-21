from pathlib import Path

from lisai.config.json_schema import write_local_config_json_schema


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "configs" / "schema" / "local-config.schema.json"
    write_local_config_json_schema(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
