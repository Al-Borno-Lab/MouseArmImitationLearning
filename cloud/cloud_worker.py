#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TRAINING_CONFIG_PATH = REPOSITORY_ROOT / "imitation_learning" / "config.yml"
TRAINING_SCRIPT = REPOSITORY_ROOT / "imitation_learning" / "train_test_record.py"
SCALING_SCRIPT = REPOSITORY_ROOT / "imitation_learning" / "scale_model.py"

MODELS_DIRECTORY = REPOSITORY_ROOT / "models"
SOURCE_MUSCLE_MODEL = MODELS_DIRECTORY / "mujoco_model_muscle.xml"
SOURCE_TORQUE_MODEL = MODELS_DIRECTORY / "mujoco_model_torque.xml"
SCALED_MUSCLE_MODEL = MODELS_DIRECTORY / "muscle.xml"
SCALED_TORQUE_MODEL = MODELS_DIRECTORY / "torque.xml"

DATA_DIRECTORY = REPOSITORY_ROOT / "cloud_data"
JOBS_DIRECTORY = REPOSITORY_ROOT / "jobs"


def run_command(command: list[str], *, dry_run: bool = False) -> None:
    print("$", " ".join(command))

    if dry_run:
        return

    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def load_assignment(path: str | Path) -> dict[str, Any]:
    assignment_path = Path(path)

    if not assignment_path.is_file():
        raise FileNotFoundError(f"Assignment file not found: {assignment_path}")

    with assignment_path.open("r", encoding="utf-8") as file:
        assignment = json.load(file)

    if not isinstance(assignment, dict):
        raise ValueError("The assignment JSON must contain an object at its root.")

    for required_key in ("bucket", "data", "base_config", "jobs"):
        if required_key not in assignment:
            raise ValueError(
                f"Assignment JSON is missing required field: {required_key}"
            )

    if not isinstance(assignment["data"], dict):
        raise ValueError("'data' must be an object.")

    if not isinstance(assignment["base_config"], dict):
        raise ValueError("'base_config' must be an object.")

    if not isinstance(assignment["jobs"], list):
        raise ValueError("'jobs' must be a list.")

    return assignment


def set_nested_value(
    config: dict[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    parts = dotted_key.split(".")
    current = config

    for part in parts[:-1]:
        child = current.get(part)

        if child is None:
            child = {}
            current[part] = child

        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot set '{dotted_key}': '{part}' is not a mapping."
            )

        current = child

    current[parts[-1]] = value


def write_training_config(
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    TRAINING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TRAINING_CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def hugging_face_repository_id(repository: str) -> str:
    repository = repository.rstrip("/")

    marker = "huggingface.co/datasets/"
    if marker in repository:
        return repository.split(marker, 1)[1]

    return repository


def download_data(data_config: dict[str, Any], *, dry_run: bool) -> Path:
    repository = str(data_config.get("repository", "")).strip()
    token = str(data_config.get("token", "")).strip()

    if not repository:
        raise ValueError("'data.repository' is empty.")

    repository_id = hugging_face_repository_id(repository)

    command = [
        "hf",
        "download",
        repository_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(DATA_DIRECTORY),
    ]

    environment = os.environ.copy()
    if token:
        environment["HF_TOKEN"] = token

    print("$", " ".join(command))

    if not dry_run:
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

        max_attempts = 5

        for attempt in range(1, max_attempts + 1):
            try:
                subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    check=True,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == max_attempts:
                    raise

                wait_seconds = attempt * 30
                print(
                    f"Hugging Face download failed "
                    f"(attempt {attempt}/{max_attempts}). "
                    f"Retrying in {wait_seconds} seconds...",
                    flush=True,
                )
                time.sleep(wait_seconds)

    return DATA_DIRECTORY


def resolve_data_path(data_root: Path, configured_path: Any) -> Path | None:
    if configured_path is None:
        return None

    text = str(configured_path).strip()
    if not text:
        return None

    path = Path(text)

    if path.is_absolute():
        return path

    return data_root / path


def select_scaling_kinematics(kinematics_path: Path) -> Path:
    if kinematics_path.is_file():
        return kinematics_path

    if not kinematics_path.is_dir():
        raise FileNotFoundError(
            f"Kinematics path is neither a file nor folder: {kinematics_path}"
        )

    files = sorted(
        path
        for path in kinematics_path.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )

    if not files:
        raise FileNotFoundError(
            f"No files found under kinematics folder: {kinematics_path}"
        )

    return files[0]


def scale_models(kinematics_file: Path, *, dry_run: bool) -> None:
    run_command(
        [
            "python",
            str(SCALING_SCRIPT),
            str(SOURCE_MUSCLE_MODEL),
            str(kinematics_file),
            str(SCALED_MUSCLE_MODEL),
        ],
        dry_run=dry_run,
    )

    run_command(
        [
            "python",
            str(SCALING_SCRIPT),
            str(SOURCE_TORQUE_MODEL),
            str(kinematics_file),
            str(SCALED_TORQUE_MODEL),
        ],
        dry_run=dry_run,
    )


def prepare_firing_rates(
    data_config: dict[str, Any],
    data_root: Path,
    *,
    dry_run: bool,
) -> Path | None:
    firing_rates = resolve_data_path(data_root, data_config.get("firing_rates"))
    if firing_rates is not None:
        return firing_rates

    spikes_config = data_config.get("spikes")
    if not isinstance(spikes_config, dict):
        return None

    spike_input = resolve_data_path(
        data_root,
        spikes_config.get("folder_or_file"),
    )

    if spike_input is None:
        return None

    spike_output = resolve_data_path(
        data_root,
        spikes_config.get("output_folder_or_file"),
    )

    if spike_output is None:
        raise ValueError(
            "'data.spikes.output_folder_or_file' is required when spikes are used."
        )

    num_steps = spikes_config.get("num_steps")
    step_dt = spikes_config.get("step_dt")

    if num_steps is None or step_dt is None:
        raise ValueError(
            "'data.spikes.num_steps' and 'data.spikes.step_dt' are required "
            "when spikes are used."
        )

    run_command(
        [
            "python",
            str(REPOSITORY_ROOT / "analysis" / "firing_rate_estimation.py"),
            str(spike_input),
            str(spike_output),
            "--num-steps",
            str(num_steps),
            "--step-dt",
            str(step_dt),
        ],
        dry_run=dry_run,
    )

    return spike_output


def selected_model_path(model_value: Any) -> Path:
    model_name = str(model_value).strip().lower()

    if model_name == "muscle":
        return SCALED_MUSCLE_MODEL

    if model_name == "torque":
        return SCALED_TORQUE_MODEL

    raise ValueError(
        "The job parameter 'environment.model' must be 'muscle' or 'torque', "
        f"not {model_value!r}."
    )


def parameter_folder_name(parameter_name: str, value: Any) -> str:
    short_name = parameter_name.split(".")[-1]
    normalized_value = str(value).strip().lower()
    return f"{short_name}-{normalized_value}"


def get_sweep_parameters(job: dict[str, Any]) -> dict[str, Any]:
    sweep_parameters = job.get("sweep_parameters")

    if not isinstance(sweep_parameters, dict) or not sweep_parameters:
        raise ValueError(
            "Every job must contain a non-empty 'sweep_parameters' object."
        )

    return sweep_parameters


def build_job_relative_path(job: dict[str, Any]) -> Path:
    sweep_parameters = get_sweep_parameters(job)

    return Path(
        *[
            parameter_folder_name(parameter_name, value)
            for parameter_name, value in sweep_parameters.items()
        ]
    )


def build_job_config(
    base_config: dict[str, Any],
    job: dict[str, Any],
    kinematics_path: Path,
) -> tuple[dict[str, Any], Path]:
    job_id = str(job.get("id", "")).strip()
    sweep_parameters = get_sweep_parameters(job)
    parameters = job.get("parameters", {})

    if not job_id:
        raise ValueError("Every job must have a non-empty 'id'.")

    if not isinstance(parameters, dict):
        raise ValueError(f"Job '{job_id}' has no valid 'parameters' object.")

    job_config = copy.deepcopy(base_config)

    for parameter_name, parameter_value in sweep_parameters.items():
        set_nested_value(job_config, parameter_name, parameter_value)

    for parameter_name, parameter_value in parameters.items():
        if parameter_name in sweep_parameters:
            raise ValueError(
                f"Job '{job_id}' duplicates sweep parameter "
                f"'{parameter_name}' in 'parameters'."
            )

        set_nested_value(job_config, parameter_name, parameter_value)

    model_value = sweep_parameters.get(
        "environment.model",
        job_config.get("environment", {}).get("model"),
    )
    model_path = selected_model_path(model_value)

    output_folder = JOBS_DIRECTORY / job_id

    set_nested_value(job_config, "general.name", job_id)
    set_nested_value(job_config, "general.folder", str(JOBS_DIRECTORY))
    set_nested_value(job_config, "general.mode", "train")
    set_nested_value(job_config, "environment.kinematics", str(kinematics_path))
    set_nested_value(job_config, "environment.model", str(model_path))

    return job_config, output_folder


def upload_job(
    output_folder: Path,
    bucket: str,
    job: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    relative_path = build_job_relative_path(job)
    destination = f"gs://{bucket.rstrip('/')}/{relative_path.as_posix()}"

    run_command(
        [
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            str(output_folder),
            destination,
        ],
        dry_run=dry_run,
    )


def run_job(
    base_config: dict[str, Any],
    job: dict[str, Any],
    kinematics_path: Path,
    bucket: str,
    *,
    dry_run: bool,
) -> None:
    job_config, output_folder = build_job_config(
        base_config,
        job,
        kinematics_path,
    )
    job_id = job["id"]

    print(f"\n=== {job_id}: train config ===")
    write_training_config(
        job_config,
        dry_run=dry_run,
    )

    print(f"=== {job_id}: train ===")
    run_command(
        ["python", str(TRAINING_SCRIPT), "--disable-progressbar"],
        dry_run=dry_run,
    )

    set_nested_value(job_config, "general.mode", "record")

    print(f"\n=== {job_id}: record config ===")
    write_training_config(
        job_config,
        dry_run=dry_run,
    )

    print(f"=== {job_id}: record ===")
    run_command(
        ["python", str(TRAINING_SCRIPT)],
        dry_run=dry_run,
    )

    print(f"\n=== {job_id}: upload ===")
    upload_job(
        output_folder,
        bucket,
        job,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cloud training jobs assigned to one VM."
    )
    parser.add_argument(
        "assignment",
        help="Path to this VM's assignment JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the assignment and print generated configs and commands "
            "without downloading, scaling, training, recording, uploading, "
            "or modifying imitation_learning/config.yml."
        ),
    )
    args = parser.parse_args()

    assignment = load_assignment(args.assignment)

    bucket = str(assignment["bucket"]).strip()
    if not bucket:
        raise ValueError("'bucket' is empty.")

    data_config = assignment["data"]
    base_config = assignment["base_config"]
    jobs = assignment["jobs"]

    data_root = download_data(data_config, dry_run=args.dry_run)

    kinematics_path = resolve_data_path(
        data_root,
        data_config.get("kinematics"),
    )
    if kinematics_path is None:
        raise ValueError("'data.kinematics' is empty.")

    if args.dry_run:
        scaling_kinematics = kinematics_path
    else:
        scaling_kinematics = select_scaling_kinematics(kinematics_path)

    scale_models(scaling_kinematics, dry_run=args.dry_run)

    firing_rates = prepare_firing_rates(
        data_config,
        data_root,
        dry_run=args.dry_run,
    )

    if firing_rates is None:
        print("No firing rates or spikes configured; neural processing is disabled.")
    else:
        print(f"Firing rates available at: {firing_rates}")

    if not jobs:
        raise ValueError("The assignment contains no jobs.")

    for job in jobs:
        run_job(
            base_config,
            job,
            kinematics_path,
            bucket,
            dry_run=args.dry_run,
        )

    print(f"\nCompleted {len(jobs)} assigned jobs.")


if __name__ == "__main__":
    main()
