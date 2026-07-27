#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

import yaml


CLOUD_CONFIG_PATH = Path(__file__).with_name("config.yml")
BASE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "imitation_learning"
    / "config.yml"
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at its root: {path}")

    return data


def normalize_sweep_option(
    parameter_name: str,
    option: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return a sweep value and any parameters attached to that value."""
    if not isinstance(option, dict):
        return option, {}

    if "value" not in option:
        raise ValueError(
            f"Sweep entry for '{parameter_name}' is a mapping but has no 'value'."
        )

    parameters = option.get("parameters", {})
    if parameters is None:
        parameters = {}

    if not isinstance(parameters, dict):
        raise ValueError(
            f"'parameters' for '{parameter_name}={option['value']}' "
            "must be a mapping."
        )

    return option["value"], dict(parameters)


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-") or "empty"


def generate_job_list(cloud_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand every combination listed directly under the jobs section."""
    sweep_parameters = cloud_config.get("jobs")

    if not isinstance(sweep_parameters, dict) or not sweep_parameters:
        raise ValueError("'jobs' must be a non-empty mapping of sweep parameters.")

    parameter_names = list(sweep_parameters)
    normalized_options: list[list[tuple[Any, dict[str, Any]]]] = []

    for parameter_name, options in sweep_parameters.items():
        if not isinstance(options, list) or not options:
            raise ValueError(
                f"Sweep parameter '{parameter_name}' must contain a non-empty list."
            )

        normalized_options.append(
            [
                normalize_sweep_option(parameter_name, option)
                for option in options
            ]
        )

    jobs: list[dict[str, Any]] = []

    for index, combination in enumerate(
        itertools.product(*normalized_options),
        start=1,
    ):
        sweep_values: dict[str, Any] = {}
        parameters: dict[str, Any] = {}
        id_parts: list[str] = []

        for parameter_name, (value, extra_parameters) in zip(
            parameter_names,
            combination,
        ):
            sweep_values[parameter_name] = value
            id_parts.append(f"{parameter_name.split('.')[-1]}-{slug(value)}")

            for extra_name, extra_value in extra_parameters.items():
                if extra_name in sweep_values:
                    raise ValueError(
                        f"Conditional parameter '{extra_name}' duplicates a "
                        f"sweep parameter while generating job {index}."
                    )

                if extra_name in parameters and parameters[extra_name] != extra_value:
                    raise ValueError(
                        "Conflicting conditional values while generating "
                        f"job {index} for '{extra_name}'."
                    )

                parameters[extra_name] = extra_value

        jobs.append(
            {
                "id": f"job-{index:04d}-" + "-".join(id_parts),
                "sweep_parameters": sweep_values,
                "parameters": parameters,
            }
        )

    return jobs


def generate_assignment(
    cloud_config_path: str | Path = CLOUD_CONFIG_PATH,
    base_config_path: str | Path = BASE_CONFIG_PATH,
) -> dict[str, Any]:
    """Build the complete payload that submit_jobs sends to a worker VM."""
    cloud_config = load_yaml(cloud_config_path)
    base_config = load_yaml(base_config_path)

    cloud = cloud_config.get("cloud")
    if not isinstance(cloud, dict):
        raise ValueError("Cloud config must contain a 'cloud' section.")

    bucket = cloud.get("bucket")
    if not bucket:
        raise ValueError("'cloud.bucket' must be filled in.")

    data = cloud_config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Cloud config must contain a 'data' section.")

    return {
        "bucket": bucket,
        "data": data,
        "base_config": base_config,
        "jobs": generate_job_list(cloud_config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the complete cloud-worker assignment JSON."
    )
    parser.add_argument(
        "--config",
        default=str(CLOUD_CONFIG_PATH),
        help="Path to cloud/config.yml.",
    )
    parser.add_argument(
        "--base-config",
        default=str(BASE_CONFIG_PATH),
        help="Path to imitation_learning/config.yml.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output file.",
    )
    args = parser.parse_args()

    assignment = generate_assignment(args.config, args.base_config)
    output = json.dumps(assignment, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Generated {len(assignment['jobs'])} jobs: {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
