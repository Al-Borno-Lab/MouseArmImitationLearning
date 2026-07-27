#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

from generate_jobs import generate_assignment


CLOUD_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CLOUD_DIRECTORY / "config.yml"
DEFAULT_BASE_CONFIG_PATH = (
    CLOUD_DIRECTORY.parent / "imitation_learning" / "config.yml"
)


def run_command(command: list[str], *, capture_output: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )
    return result.stdout if capture_output else ""


def parameter_folder_name(parameter_name: str, value: Any) -> str:
    short_name = parameter_name.split(".")[-1]
    normalized_value = str(value).strip().lower()
    return f"{short_name}-{normalized_value}"


def job_result_prefix(job: dict[str, Any]) -> str:
    sweep_parameters = job.get("sweep_parameters")

    if not isinstance(sweep_parameters, dict) or not sweep_parameters:
        raise ValueError(
            f"Job {job.get('id', '<unknown>')} has no valid "
            "'sweep_parameters' object."
        )

    return "/".join(
        parameter_folder_name(parameter_name, value)
        for parameter_name, value in sweep_parameters.items()
    ) + "/"


def list_bucket_objects(bucket: str, project_id: str) -> set[str]:
    bucket = bucket.removeprefix("gs://").rstrip("/")

    output = run_command(
        [
            "gcloud",
            "storage",
            "ls",
            "--recursive",
            f"gs://{bucket}",
            "--project",
            project_id,
        ],
        capture_output=True,
    )

    prefix = f"gs://{bucket}/"
    objects: set[str] = set()

    for line in output.splitlines():
        line = line.strip()

        if not line.startswith(prefix):
            continue

        relative_path = line[len(prefix):]

        if relative_path and not relative_path.endswith("/"):
            objects.add(relative_path)

    return objects


def completed_job_ids(
    jobs: list[dict[str, Any]],
    objects: set[str],
) -> set[str]:
    completed: set[str] = set()

    for job in jobs:
        job_id = str(job.get("id", "")).strip()
        prefix = job_result_prefix(job)

        if any(object_name.startswith(prefix) for object_name in objects):
            completed.add(job_id)

    return completed


def print_progress(completed: int, total: int) -> None:
    percentage = 100.0 * completed / total if total else 100.0
    print(
        f"\rCompleted: {completed}/{total} jobs "
        f"({percentage:.1f}%)",
        end="",
        flush=True,
    )


def download_results(
    *,
    bucket: str,
    project_id: str,
    download_directory: Path,
) -> None:
    bucket = bucket.removeprefix("gs://").rstrip("/")
    download_directory.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading results to: {download_directory}")

    run_command(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            "--exclude",
            r"^assignments/.*",
            f"gs://{bucket}",
            str(download_directory),
            "--project",
            project_id,
        ]
    )

    print("Download complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor completion of the configured cloud workload and "
            "optionally download all results when it reaches 100%."
        )
    )
    parser.add_argument(
        "download_directory",
        nargs="?",
        help=(
            "Optional local directory. If provided, results are downloaded "
            "there after all jobs are complete."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to cloud/config.yml.",
    )
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_BASE_CONFIG_PATH),
        help="Path to imitation_learning/config.yml.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between checks. Default: 60.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once instead of continuing until completion.",
    )
    args = parser.parse_args()

    if args.interval < 1:
        raise ValueError("--interval must be at least 1 second.")

    assignment = generate_assignment(
        Path(args.config).resolve(),
        Path(args.base_config).resolve(),
    )

    bucket = str(assignment["bucket"]).strip()
    jobs = assignment["jobs"]

    if not bucket:
        raise ValueError("'cloud.bucket' is empty.")

    if not jobs:
        raise ValueError("The configured workload contains no jobs.")

    # project_id belongs to cloud/config.yml but is not included in the
    # worker assignment, so read it directly here.
    import yaml

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        cloud_config = yaml.safe_load(file)

    project_id = str(
        cloud_config.get("cloud", {}).get("project_id", "")
    ).strip()

    if not project_id:
        raise ValueError("'cloud.project_id' is empty.")

    total = len(jobs)
    previous_completed = -1

    while True:
        try:
            objects = list_bucket_objects(bucket, project_id)
        except subprocess.CalledProcessError as error:
            print()
            raise RuntimeError(
                "Could not list the bucket. Check your gcloud login, "
                "project, bucket name, and permissions."
            ) from error

        completed_ids = completed_job_ids(jobs, objects)
        completed = len(completed_ids)

        if completed != previous_completed:
            print_progress(completed, total)
            previous_completed = completed

        if completed == total:
            print()

            if args.download_directory:
                download_results(
                    bucket=bucket,
                    project_id=project_id,
                    download_directory=Path(
                        args.download_directory
                    ).expanduser().resolve(),
                )

            return

        if args.once:
            print()
            return

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
