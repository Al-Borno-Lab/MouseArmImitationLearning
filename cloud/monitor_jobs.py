#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_jobs import generate_assignment


CLOUD_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CLOUD_DIRECTORY / "config.yml"
DEFAULT_BASE_CONFIG_PATH = (
    CLOUD_DIRECTORY.parent / "imitation_learning" / "config.yml"
)
DEFAULT_LOG_DIRECTORY = CLOUD_DIRECTORY / "logs"


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



def metadata_value(instance: dict[str, Any], key: str) -> str:
    metadata = instance.get("metadata", {})
    items = metadata.get("items", []) if isinstance(metadata, dict) else []

    for item in items:
        if isinstance(item, dict) and item.get("key") == key:
            return str(item.get("value", "")).strip()

    return ""


def list_worker_instances(
    *,
    project_id: str,
    vm_name_prefix: str,
) -> list[dict[str, Any]]:
    output = run_command(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project_id,
            "--filter",
            f"name~'^{vm_name_prefix}-'",
            "--format=json(name,zone,status,metadata.items)",
        ],
        capture_output=True,
    )

    instances = json.loads(output or "[]")

    if not isinstance(instances, list):
        raise RuntimeError("Unexpected response while listing VM instances.")

    return [
        instance
        for instance in instances
        if isinstance(instance, dict)
    ]


def load_assignment_jobs(
    *,
    bucket: str,
    assignment_index: str,
    project_id: str,
) -> list[dict[str, Any]]:
    bucket = bucket.removeprefix("gs://").rstrip("/")

    output = run_command(
        [
            "gcloud",
            "storage",
            "cat",
            f"gs://{bucket}/assignments/{assignment_index}.json",
            "--project",
            project_id,
        ],
        capture_output=True,
    )

    assignment = json.loads(output)
    jobs = assignment.get("jobs", [])

    if not isinstance(jobs, list):
        raise ValueError(
            f"Assignment {assignment_index} does not contain a valid jobs list."
        )

    return [
        job
        for job in jobs
        if isinstance(job, dict)
    ]


def startup_service_state(
    *,
    vm_name: str,
    zone: str,
    project_id: str,
) -> str:
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            "--zone",
            zone,
            "--project",
            project_id,
            "--quiet",
            "--command",
            (
                "systemctl is-active google-startup-scripts.service "
                "2>/dev/null || true"
            ),
        ],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        return "unreachable"

    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else "unknown"


def retrieve_startup_log(
    *,
    vm_name: str,
    zone: str,
    project_id: str,
) -> str:
    journal = subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            "--zone",
            zone,
            "--project",
            project_id,
            "--quiet",
            "--command",
            (
                "sudo journalctl "
                "-u google-startup-scripts.service "
                "--no-pager -n 250"
            ),
        ],
        text=True,
        capture_output=True,
    )

    if journal.returncode == 0 and journal.stdout.strip():
        return journal.stdout

    serial = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "get-serial-port-output",
            vm_name,
            "--zone",
            zone,
            "--project",
            project_id,
        ],
        text=True,
        capture_output=True,
    )

    if serial.returncode == 0 and serial.stdout.strip():
        return serial.stdout

    return (
        "Could not retrieve the startup journal or serial-port output.\n\n"
        f"SSH stderr:\n{journal.stderr.strip()}\n\n"
        f"Serial stderr:\n{serial.stderr.strip()}\n"
    )


def write_vm_error_log(
    *,
    log_directory: Path,
    vm_name: str,
    zone: str,
    vm_status: str,
    startup_state: str,
    assignment_index: str,
    incomplete_jobs: list[dict[str, Any]],
    startup_log: str,
) -> Path:
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"{vm_name}.log"

    incomplete_job_ids = [
        str(job.get("id", "<unknown>"))
        for job in incomplete_jobs
    ]

    header = [
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"vm_name: {vm_name}",
        f"zone: {zone}",
        f"vm_status: {vm_status}",
        f"startup_service_state: {startup_state}",
        f"assignment_index: {assignment_index}",
        f"incomplete_job_count: {len(incomplete_job_ids)}",
        "incomplete_job_ids:",
        *[
            f"  - {job_id}"
            for job_id in incomplete_job_ids
        ],
        "",
        "startup_log:",
        "",
    ]

    log_path.write_text(
        "\n".join(header) + startup_log.rstrip() + "\n",
        encoding="utf-8",
    )
    return log_path


def delete_vm(
    *,
    vm_name: str,
    zone: str,
    project_id: str,
) -> None:
    run_command(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            vm_name,
            "--zone",
            zone,
            "--project",
            project_id,
            "--quiet",
        ]
    )


def check_vm_failures(
    *,
    bucket: str,
    project_id: str,
    vm_name_prefix: str,
    completed_ids: set[str],
    log_directory: Path,
    handled_vms: set[str],
    reachable_vms: set[str],
    unreachable_counts: dict[str, int],
    max_unreachable_checks: int,
) -> None:
    instances = list_worker_instances(
        project_id=project_id,
        vm_name_prefix=vm_name_prefix,
    )

    for instance in instances:
        vm_name = str(instance.get("name", "")).strip()
        zone = str(instance.get("zone", "")).rsplit("/", 1)[-1]
        vm_status = str(instance.get("status", "UNKNOWN")).strip()
        assignment_index = metadata_value(
            instance,
            "assignment-index",
        )

        if (
            not vm_name
            or not zone
            or not assignment_index
            or vm_name in handled_vms
        ):
            continue

        try:
            assigned_jobs = load_assignment_jobs(
                bucket=bucket,
                assignment_index=assignment_index,
                project_id=project_id,
            )
        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            ValueError,
        ):
            continue

        incomplete_jobs = [
            job
            for job in assigned_jobs
            if str(job.get("id", "")).strip()
            not in completed_ids
        ]

        if not incomplete_jobs:
            continue

        if vm_status == "RUNNING":
            startup_state = startup_service_state(
                vm_name=vm_name,
                zone=zone,
                project_id=project_id,
            )
        else:
            startup_state = "not-running"

        if startup_state == "unreachable":
            unreachable_counts[vm_name] = (
                unreachable_counts.get(vm_name, 0) + 1
            )

            if unreachable_counts[vm_name] == 1:
                print(f"\nWaiting for VM SSH: {vm_name}")

            if unreachable_counts[vm_name] < max_unreachable_checks:
                continue

            print(
                f"\nVM remained unreachable for "
                f"{unreachable_counts[vm_name]} checks: {vm_name}"
            )
        else:
            unreachable_counts.pop(vm_name, None)

            if vm_name not in reachable_vms:
                print(
                    f"\nVM reachable; startup monitoring active: "
                    f"{vm_name}"
                )
                reachable_vms.add(vm_name)

        if startup_state in {"active", "activating"}:
            continue

        startup_log = retrieve_startup_log(
            vm_name=vm_name,
            zone=zone,
            project_id=project_id,
        )

        log_path = write_vm_error_log(
            log_directory=log_directory,
            vm_name=vm_name,
            zone=zone,
            vm_status=vm_status,
            startup_state=startup_state,
            assignment_index=assignment_index,
            incomplete_jobs=incomplete_jobs,
            startup_log=startup_log,
        )

        print(
            f"\nVM died: {vm_name} "
            f"({len(incomplete_jobs)} incomplete jobs)"
        )
        print(f"Startup error log saved to: {log_path}")
        print(f"Deleting VM: {vm_name}")

        try:
            delete_vm(
                vm_name=vm_name,
                zone=zone,
                project_id=project_id,
            )
        except subprocess.CalledProcessError:
            print(
                f"Failed to delete VM automatically: {vm_name}"
            )
        else:
            print(f"Deleted VM: {vm_name}")

        handled_vms.add(vm_name)


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
        "--log-directory",
        default=str(DEFAULT_LOG_DIRECTORY),
        help="Directory for failed VM logs. Default: cloud/logs.",
    )
    parser.add_argument(
        "--vm-name-prefix",
        default="mouse-arm-worker",
        help="Prefix used by worker VM names.",
    )
    parser.add_argument(
        "--max-unreachable-checks",
        type=int,
        default=10,
        help=(
            "Consecutive failed SSH checks before treating a VM as dead. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once instead of continuing until completion.",
    )
    args = parser.parse_args()

    if args.interval < 1:
        raise ValueError("--interval must be at least 1 second.")

    if args.max_unreachable_checks < 1:
        raise ValueError(
            "--max-unreachable-checks must be at least 1."
        )

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
    handled_vms: set[str] = set()
    reachable_vms: set[str] = set()
    unreachable_counts: dict[str, int] = {}
    log_directory = Path(
        args.log_directory
    ).expanduser().resolve()

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

        try:
            check_vm_failures(
                bucket=bucket,
                project_id=project_id,
                vm_name_prefix=args.vm_name_prefix,
                completed_ids=completed_ids,
                log_directory=log_directory,
                handled_vms=handled_vms,
                reachable_vms=reachable_vms,
                unreachable_counts=unreachable_counts,
                max_unreachable_checks=args.max_unreachable_checks,
            )
        except subprocess.CalledProcessError as error:
            print(
                "\nWarning: VM health check failed: "
                f"{error}"
            )

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
