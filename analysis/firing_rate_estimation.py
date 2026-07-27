import argparse
import json
from pathlib import Path

import numpy as np
import quantities as pq
import neo
from elephant.statistics import instantaneous_rate
from elephant.kernels import GaussianKernel


def estimate_firing_rates_from_json(
    json_path,
    num_steps,
    step_dt,
    sigma=0.05,
    time_unit=pq.s,
):
    """
    Estimate firing rates from spike-time JSON.

    Input JSON format:
        [
            [spike_times_for_neuron_0],
            [spike_times_for_neuron_1],
            ...
        ]

    Returns:
        rates: shape [N_neurons, num_steps], in Hz
    """

    json_path = Path(json_path)

    with json_path.open("r") as f:
        spike_time_lists = json.load(f)

    start_time = 0.0
    end_time = num_steps * step_dt

    t_start = start_time * time_unit
    t_stop = end_time * time_unit
    sampling_period = step_dt * time_unit
    kernel = GaussianKernel(sigma=sigma * time_unit)

    all_rates = []

    for spike_times in spike_time_lists:
        spike_times = np.asarray(spike_times, dtype=float)

        filtered_spike_times = spike_times[
            (spike_times >= start_time) & (spike_times <= end_time)
        ]

        st = neo.SpikeTrain(
            filtered_spike_times * time_unit,
            t_start=t_start,
            t_stop=t_stop,
        )

        rate_signal = instantaneous_rate(
            st,
            sampling_period=sampling_period,
            kernel=kernel,
            t_start=t_start,
            t_stop=t_stop,
        )

        rate_hz = rate_signal.rescale(pq.Hz).magnitude.squeeze()

        # Force exact timestep count.
        rate_hz = rate_hz[:num_steps]

        if rate_hz.shape[0] < num_steps:
            pad_amount = num_steps - rate_hz.shape[0]
            rate_hz = np.pad(rate_hz, (0, pad_amount), mode="edge")

        all_rates.append(rate_hz)

    rates = np.stack(all_rates, axis=0)

    return rates


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f)


def process_one_file(
    input_json,
    output_json,
    num_steps,
    step_dt,
    sigma,
):
    rates = estimate_firing_rates_from_json(
        json_path=input_json,
        num_steps=num_steps,
        step_dt=step_dt,
        sigma=sigma,
    )

    save_json(output_json, rates.tolist())

    print(f"Saved firing rates to: {output_json}")
    print(f"rates shape: {rates.shape}")


def process_file_or_folder(
    spike_times_json,
    output_dir,
    num_steps,
    step_dt,
    sigma,
):
    spike_times_json = Path(spike_times_json)
    output_dir = Path(output_dir)

    if spike_times_json.is_file():
        output_json = output_dir / spike_times_json.name

        process_one_file(
            input_json=spike_times_json,
            output_json=output_json,
            num_steps=num_steps,
            step_dt=step_dt,
            sigma=sigma,
        )

    elif spike_times_json.is_dir():
        for input_json in spike_times_json.rglob("*.json"):
            relative_path = input_json.relative_to(spike_times_json)
            output_json = output_dir / relative_path

            process_one_file(
                input_json=input_json,
                output_json=output_json,
                num_steps=num_steps,
                step_dt=step_dt,
                sigma=sigma,
            )

    else:
        raise FileNotFoundError(f"Could not find: {spike_times_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate firing rates from spike-time JSON files."
    )

    parser.add_argument(
        "spike_times_json",
        help="Input spike-time JSON file or folder of JSON files.",
    )

    parser.add_argument(
        "output_dir",
        help="Output folder where firing-rate JSON files will be saved.",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of output timesteps. Start time is always 0.",
    )

    parser.add_argument(
        "--step-dt",
        type=float,
        required=True,
        help="Timestep spacing in seconds.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Gaussian kernel sigma in seconds. Default 0.05.",
    )

    args = parser.parse_args()

    process_file_or_folder(
        spike_times_json=args.spike_times_json,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        step_dt=args.step_dt,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()