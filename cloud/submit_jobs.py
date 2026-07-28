#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from generate_jobs import generate_assignment


CLOUD_DIRECTORY = Path(__file__).resolve().parent
CLOUD_CONFIG_PATH = CLOUD_DIRECTORY / "config.yml"
BASE_CONFIG_PATH = CLOUD_DIRECTORY.parent / "imitation_learning" / "config.yml"
STARTUP_SCRIPT_PATH = CLOUD_DIRECTORY / "startup.sh"
ASSIGNMENTS_DIRECTORY = CLOUD_DIRECTORY / "assignments"

DEFAULT_IMAGE_FAMILY = "common-cu129-ubuntu-2404-nvidia-580"
DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at its root: {path}")

    return data


def run_command(
    command: list[str],
    *,
    capture_output: bool = False,
    dry_run: bool = False,
) -> str:
    print("$", " ".join(command))

    if dry_run:
        return ""

    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )

    return result.stdout if capture_output else ""


def get_sweep_parameters(job: dict[str, Any]) -> dict[str, Any]:
    sweep_parameters = job.get("sweep_parameters")

    if not isinstance(sweep_parameters, dict) or not sweep_parameters:
        raise ValueError(
            f"Job {job.get('id', '<unknown>')} has no valid "
            "'sweep_parameters' object."
        )

    return sweep_parameters


def assignment_balance_score(
    assignments: list[list[dict[str, Any]]],
    parameter_names: list[str],
) -> tuple[int, int, int]:
    load_counts = [len(group) for group in assignments]
    load_range = max(load_counts) - min(load_counts)

    worst_value_range = 0
    total_value_range = 0

    for parameter_name in parameter_names:
        values = {
            get_sweep_parameters(job)[parameter_name]
            for group in assignments
            for job in group
        }

        for value in values:
            counts = [
                sum(
                    get_sweep_parameters(job)[parameter_name] == value
                    for job in group
                )
                for group in assignments
            ]
            value_range = max(counts) - min(counts)
            worst_value_range = max(worst_value_range, value_range)
            total_value_range += value_range

    return load_range, worst_value_range, total_value_range


def greedy_balanced_split(
    jobs: list[dict[str, Any]],
    number_of_vms: int,
    *,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if number_of_vms < 1:
        raise ValueError("'compute.number_of_vms' must be at least 1.")

    if number_of_vms > len(jobs):
        raise ValueError(
            "'compute.number_of_vms' cannot exceed the number of generated jobs."
        )

    parameter_names = list(get_sweep_parameters(jobs[0]))
    maximum_size = math.ceil(len(jobs) / number_of_vms)

    shuffled_jobs = list(jobs)
    random.Random(seed).shuffle(shuffled_jobs)

    assignments: list[list[dict[str, Any]]] = [
        [] for _ in range(number_of_vms)
    ]
    counts: list[dict[str, Counter[Any]]] = [
        {name: Counter() for name in parameter_names}
        for _ in range(number_of_vms)
    ]

    global_counts: dict[str, Counter[Any]] = {
        name: Counter(
            get_sweep_parameters(job)[name]
            for job in jobs
        )
        for name in parameter_names
    }

    target_counts: dict[str, dict[Any, float]] = {
        name: {
            value: count / number_of_vms
            for value, count in value_counts.items()
        }
        for name, value_counts in global_counts.items()
    }

    for job in shuffled_jobs:
        sweep = get_sweep_parameters(job)
        candidates = [
            vm_index
            for vm_index, group in enumerate(assignments)
            if len(group) < maximum_size
        ]

        best_vm: int | None = None
        best_score: tuple[float, int, int] | None = None

        for vm_index in candidates:
            imbalance = 0.0

            for parameter_name, value in sweep.items():
                old_count = counts[vm_index][parameter_name][value]
                new_count = old_count + 1
                target = target_counts[parameter_name][value]

                imbalance += (
                    (new_count - target) ** 2
                    - (old_count - target) ** 2
                )

            score = (
                imbalance,
                len(assignments[vm_index]),
                vm_index,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_vm = vm_index

        if best_vm is None:
            raise RuntimeError("No VM assignment slot was available.")

        assignments[best_vm].append(job)

        for parameter_name, value in sweep.items():
            counts[best_vm][parameter_name][value] += 1

    return assignments


def improve_assignment_by_swapping(
    assignments: list[list[dict[str, Any]]],
    parameter_names: list[str],
) -> list[list[dict[str, Any]]]:
    best_score = assignment_balance_score(assignments, parameter_names)

    while True:
        improved = False

        for first_vm in range(len(assignments)):
            for second_vm in range(first_vm + 1, len(assignments)):
                for first_job_index in range(len(assignments[first_vm])):
                    for second_job_index in range(len(assignments[second_vm])):
                        assignments[first_vm][first_job_index], assignments[second_vm][second_job_index] = (
                            assignments[second_vm][second_job_index],
                            assignments[first_vm][first_job_index],
                        )

                        score = assignment_balance_score(
                            assignments,
                            parameter_names,
                        )

                        if score < best_score:
                            best_score = score
                            improved = True
                            break

                        assignments[first_vm][first_job_index], assignments[second_vm][second_job_index] = (
                            assignments[second_vm][second_job_index],
                            assignments[first_vm][first_job_index],
                        )

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            return assignments


def split_jobs_balanced(
    jobs: list[dict[str, Any]],
    number_of_vms: int,
    *,
    attempts: int = 200,
) -> list[list[dict[str, Any]]]:
    parameter_names = list(get_sweep_parameters(jobs[0]))
    best_assignments: list[list[dict[str, Any]]] | None = None
    best_score: tuple[int, int, int] | None = None

    for seed in range(attempts):
        assignments = greedy_balanced_split(
            jobs,
            number_of_vms,
            seed=seed,
        )
        score = assignment_balance_score(
            assignments,
            parameter_names,
        )

        if best_score is None or score < best_score:
            best_score = score
            best_assignments = assignments

            if score == (0, 0, 0):
                break

    if best_assignments is None:
        raise RuntimeError("Failed to split jobs.")

    return improve_assignment_by_swapping(
        best_assignments,
        parameter_names,
    )


def print_balance_summary(
    assignments: list[list[dict[str, Any]]],
) -> None:
    parameter_names = list(
        get_sweep_parameters(assignments[0][0])
    )

    print("\nJob distribution:")
    for vm_index, jobs in enumerate(assignments, start=1):
        print(f"  VM {vm_index}: {len(jobs)} jobs")

    for parameter_name in parameter_names:
        print(f"\n  {parameter_name}:")

        all_values = []
        for group in assignments:
            for job in group:
                value = get_sweep_parameters(job)[parameter_name]
                if value not in all_values:
                    all_values.append(value)

        for value in all_values:
            counts = [
                sum(
                    get_sweep_parameters(job)[parameter_name] == value
                    for job in group
                )
                for group in assignments
            ]
            print(f"    {value}: {counts}")


def list_resource_zones(
    *,
    project_id: str,
    resource_group: str,
    resource_name: str,
    region: str,
    dry_run: bool,
) -> set[str]:
    command = [
        "gcloud",
        "compute",
        resource_group,
        "list",
        "--project",
        project_id,
        "--filter",
        f"name={resource_name}",
        "--format=json(name,zone)",
    ]

    output = run_command(
        command,
        capture_output=True,
        dry_run=dry_run,
    )

    if dry_run:
        return {f"{region}-DRY-RUN"}

    records = json.loads(output)
    zones: set[str] = set()

    for record in records:
        zone_url = str(record.get("zone", ""))
        zone = zone_url.rsplit("/", 1)[-1]

        if zone.startswith(f"{region}-"):
            zones.add(zone)

    return zones


def uses_integrated_gpu_machine(machine_type: str) -> bool:
    return machine_type.startswith("g2-")


def discover_compatible_zones(
    *,
    project_id: str,
    region: str,
    machine_type: str,
    gpu: str,
    dry_run: bool,
) -> list[str]:
    machine_zones = list_resource_zones(
        project_id=project_id,
        resource_group="machine-types",
        resource_name=machine_type,
        region=region,
        dry_run=dry_run,
    )

    if uses_integrated_gpu_machine(machine_type):
        compatible_zones = sorted(machine_zones)
    else:
        gpu_zones = list_resource_zones(
            project_id=project_id,
            resource_group="accelerator-types",
            resource_name=gpu,
            region=region,
            dry_run=dry_run,
        )
        compatible_zones = sorted(gpu_zones & machine_zones)

    if not compatible_zones:
        if uses_integrated_gpu_machine(machine_type):
            raise RuntimeError(
                f"No zones in region '{region}' support machine type "
                f"'{machine_type}'."
            )

        raise RuntimeError(
            f"No zones in region '{region}' support both GPU '{gpu}' "
            f"and machine type '{machine_type}'."
        )

    return compatible_zones


def build_worker_assignment(
    full_assignment: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    worker_assignment = copy.deepcopy(full_assignment)
    worker_assignment["jobs"] = jobs
    return worker_assignment


def write_assignment(
    assignment: dict[str, Any],
    *,
    vm_index: int,
    output_directory: Path,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    assignment_path = output_directory / f"vm-{vm_index:03d}.json"
    assignment_path.write_text(
        json.dumps(assignment, indent=2) + "\n",
        encoding="utf-8",
    )
    return assignment_path


def upload_assignment(
    *,
    assignment_path: Path,
    bucket: str,
    assignment_index: str,
    project_id: str,
    dry_run: bool,
) -> str:
    destination = (
        f"gs://{bucket.rstrip('/')}/assignments/{assignment_index}.json"
    )

    run_command(
        [
            "gcloud",
            "storage",
            "cp",
            str(assignment_path),
            destination,
            "--project",
            project_id,
        ],
        dry_run=dry_run,
    )

    return destination


def create_vm(
    *,
    vm_name: str,
    zone: str,
    assignment_bucket: str,
    assignment_index: str,
    startup_script_path: Path,
    project_id: str,
    machine_type: str,
    gpu: str,
    size_gb: int,
    image_family: str,
    image_project: str,
    dry_run: bool,
) -> bool:
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        vm_name,
        "--project",
        project_id,
        "--zone",
        zone,
        "--machine-type",
        machine_type,
    ]

    if not uses_integrated_gpu_machine(machine_type):
        command.extend(
            [
                "--accelerator",
                f"type={gpu},count=1",
            ]
        )

    command.extend(
        [
            "--maintenance-policy",
            "TERMINATE",
            "--boot-disk-size",
            f"{size_gb}GB",
            "--image-family",
            image_family,
            "--image-project",
            image_project,
            "--scopes",
            "cloud-platform",
            "--metadata-from-file",
            f"startup-script={startup_script_path}",
            "--metadata",
            (
                f"assignment-bucket={assignment_bucket},"
                f"assignment-index={assignment_index}"
            ),
        ]
    )

    print("$", " ".join(command))

    if dry_run:
        return True

    result = subprocess.run(
        command,
        text=True,
    )
    return result.returncode == 0


def create_vm_in_available_zone(
    *,
    vm_name: str,
    compatible_zones: list[str],
    starting_zone_index: int,
    assignment_bucket: str,
    assignment_index: str,
    startup_script_path: Path,
    project_id: str,
    machine_type: str,
    gpu: str,
    size_gb: int,
    image_family: str,
    image_project: str,
    dry_run: bool,
) -> str:
    ordered_zones = (
        compatible_zones[starting_zone_index:]
        + compatible_zones[:starting_zone_index]
    )

    for attempt, zone in enumerate(ordered_zones, start=1):
        print(
            f"\nAttempt {attempt}/{len(ordered_zones)}: "
            f"creating {vm_name} in {zone}"
        )

        created = create_vm(
            vm_name=vm_name,
            zone=zone,
            assignment_bucket=assignment_bucket,
            assignment_index=assignment_index,
            startup_script_path=startup_script_path,
            project_id=project_id,
            machine_type=machine_type,
            gpu=gpu,
            size_gb=size_gb,
            image_family=image_family,
            image_project=image_project,
            dry_run=dry_run,
        )

        if created:
            print(f"Created {vm_name} in {zone}")
            return zone

        print(
            f"Creation failed in {zone}; trying the next compatible zone."
        )

    raise RuntimeError(
        f"Could not create VM '{vm_name}' in any compatible zone "
        f"within the selected region. Tried: {', '.join(ordered_zones)}"
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, balance, and submit cloud training jobs to GCP VMs."
        )
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
        "--startup-script",
        default=str(STARTUP_SCRIPT_PATH),
        help="Path to cloud/startup.sh.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate assignments and print GCP commands without creating VMs."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_config_path = Path(args.base_config).resolve()
    startup_script_path = Path(args.startup_script).resolve()

    cloud_config = load_yaml(config_path)

    cloud = cloud_config.get("cloud")
    compute = cloud_config.get("compute")

    if not isinstance(cloud, dict):
        raise ValueError("Config must contain a 'cloud' section.")

    if not isinstance(compute, dict):
        raise ValueError("Config must contain a 'compute' section.")

    project_id = str(cloud.get("project_id", "")).strip()
    if not project_id:
        raise ValueError("'cloud.project_id' must be filled in.")

    number_of_vms = int(compute.get("number_of_vms", 0))
    machine_type = str(compute.get("machine_type", "")).strip()
    gpu = str(compute.get("gpu", "")).strip()
    region = str(compute.get("region", "")).strip()
    size_gb = int(compute.get("size_gb", 0))

    if not machine_type:
        raise ValueError("'compute.machine_type' must be filled in.")

    if not gpu:
        raise ValueError("'compute.gpu' must be filled in.")

    if uses_integrated_gpu_machine(machine_type) and gpu != "nvidia-l4":
        raise ValueError(
            "G2 machine types include an NVIDIA L4 GPU. Set "
            "'compute.gpu' to 'nvidia-l4'."
        )

    if not region:
        raise ValueError("'compute.region' must be filled in.")

    if size_gb < 10:
        raise ValueError("'compute.size_gb' must be at least 10.")

    image_family = str(
        compute.get("image_family", DEFAULT_IMAGE_FAMILY)
    ).strip()
    image_project = str(
        compute.get("image_project", DEFAULT_IMAGE_PROJECT)
    ).strip()
    vm_name_prefix = str(
        compute.get("vm_name_prefix", "mouse-arm-worker")
    ).strip()

    full_assignment = generate_assignment(
        config_path,
        base_config_path,
    )
    jobs = full_assignment["jobs"]

    assignments = split_jobs_balanced(
        jobs,
        number_of_vms,
    )
    print_balance_summary(assignments)

    compatible_zones = discover_compatible_zones(
        project_id=project_id,
        region=region,
        machine_type=machine_type,
        gpu=gpu,
        dry_run=args.dry_run,
    )

    print("\nCompatible zones:")
    for zone in compatible_zones:
        print(f"  {zone}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_directory = ASSIGNMENTS_DIRECTORY / run_id

    assignment_bucket = str(full_assignment["bucket"]).strip()
    if not assignment_bucket:
        raise ValueError("Generated assignment has an empty 'bucket' value.")

    for vm_index, jobs_for_vm in enumerate(assignments, start=1):
        assignment_index = f"{vm_index:03d}"

        assignment = build_worker_assignment(
            full_assignment,
            jobs_for_vm,
        )
        assignment_path = write_assignment(
            assignment,
            vm_index=vm_index,
            output_directory=output_directory,
        )

        assignment_uri = upload_assignment(
            assignment_path=assignment_path,
            bucket=assignment_bucket,
            assignment_index=assignment_index,
            project_id=project_id,
            dry_run=args.dry_run,
        )

        starting_zone_index = (
            (vm_index - 1) % len(compatible_zones)
        )
        vm_name = (
            f"{vm_name_prefix}-{run_id}-{assignment_index}"
        ).lower()[:63].rstrip("-")

        print(
            f"\nCreating {vm_name} with "
            f"{len(jobs_for_vm)} jobs"
        )
        print(f"  Assignment: {assignment_uri}")

        create_vm_in_available_zone(
            vm_name=vm_name,
            compatible_zones=compatible_zones,
            starting_zone_index=starting_zone_index,
            assignment_bucket=assignment_bucket,
            assignment_index=assignment_index,
            startup_script_path=startup_script_path,
            project_id=project_id,
            machine_type=machine_type,
            gpu=gpu,
            size_gb=size_gb,
            image_family=image_family,
            image_project=image_project,
            dry_run=args.dry_run,
        )

    print(
        f"\nPrepared {len(assignments)} VM assignments in "
        f"{output_directory}"
    )


if __name__ == "__main__":
    main()
