from pathlib import Path

from lisai.config.json_schema import write_preprocess_json_schema
from lisai.config.settings import load_default_settings


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "configs" / "schema" / "preprocess.schema.json"
    write_preprocess_json_schema(output_path, stg=load_default_settings())
    print(output_path)


if __name__ == "__main__":
    main()
