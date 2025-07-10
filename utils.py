import math
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from pathlib import Path
import json


# -- Utlity functions --
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_arrival_times(
    angle_deg, speed_of_sound=300, inter_ear_distance=0.3, source_distance=1.0
):
    """
    Calculates the arrival times of sound to two ear points given the source position in ms.
    Front = 0, right = 90, back = 180, left = 270 or -90 degrees.

    Parameters:
        angle_rad (float): Angle to the sound source in radians (0 = in front, positive = to the right).
        speed_of_sound (float): Speed of sound in the medium (e.g., 343 m/s in air).
        inter_ear_distance (float): Distance between the two ears (meters).
        source_distance (float): Distance from the center point between the ears to the sound source (meters).

    Returns:
        tuple: (arrival_time_left, arrival_time_right)
    """

    angle_rad = math.radians(angle_deg)

    # Ear positions (assuming head center at (0,0), ears on y-axis)
    left_ear = (0, -inter_ear_distance / 2)
    right_ear = (0, inter_ear_distance / 2)

    # Source position in Cartesian coordinates
    source_x = source_distance * math.cos(angle_rad)
    source_y = source_distance * math.sin(angle_rad)
    source_pos = (source_x, source_y)

    # Distance from source to each ear
    dist_left = euclidean(source_pos, left_ear)
    dist_right = euclidean(source_pos, right_ear)

    # Arrival times in ms
    time_left = dist_left / speed_of_sound * 1000
    time_right = dist_right / speed_of_sound * 1000

    return (time_left, time_right)


def binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=0, jitter_ms=0):
    cycle_length = 1000 / f_stim_Hz  # ms per period
    n_cycles = int(tmax_ms / cycle_length)
    p = f_pre_Hz / f_stim_Hz
    indices = []
    times = []
    for cycle in range(n_cycles):
        base_time = cycle * cycle_length + phase
        for axon in range(N):
            if np.random.rand() < p:
                time = base_time + np.random.normal(0, jitter_ms)
                indices.append(axon)
                times.append(time)
    return indices, times


def smooth_data(angles, max_voltages, window_size=5):
    """Smooth data using a moving average over nearby points."""
    smoothed_voltages = np.zeros_like(max_voltages)
    for i in range(len(max_voltages)):
        # Define the window range
        start = max(0, i - window_size // 2)
        end = min(len(max_voltages), i + window_size // 2 + 1)
        # Average over the window
        smoothed_voltages[i] = np.mean(max_voltages[start:end])
    return smoothed_voltages


def calculate_threshold(max_voltages, percentile=0.75):
    """Get the max voltages for N angles at one neuron. Return the ideal threshold for spikes based on a percentile of N."""
    sorted_voltages = np.sort(max_voltages)
    cutoff = int(len(sorted_voltages) * percentile)
    threshold = sorted_voltages[cutoff]
    return threshold


# -- Printing and plotting functions --
def plot_jeffress_results(
    monitors, dendrite_lengths, resolution, title="Voltage traces for somas"
):
    """Plot simulation results.

    Args:
        monitors (list): List of StateMonitor objects
        dendrite_lengths (array): Array of dendrite lengths
        resolution (int): Number of neurons
        title (str): Plot title
    """
    plt.figure(figsize=(12, 8))

    for i in range(resolution):
        # soma is compartment 1 in each neuron
        plt.plot(
            monitors[i].t / ms,
            monitors[i].v[1] / mV,
            label=f"Soma {i} (L={dendrite_lengths[i]:.1f}, R={dendrite_lengths[resolution-1-i]:.1f})",
        )

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_jeffress_simulation_info(dendrite_lengths, resolution):
    """Print simulation information.

    Args:
        dendrite_lengths (array): Array of dendrite lengths
        resolution (int): Number of neurons
    """
    print(f"Simulation completed. Plotted {resolution} soma voltage traces.")
    print("Dendrite lengths (left, right) for each neuron:")
    for i in range(resolution):
        print(
            f"  Neuron {i}: L={dendrite_lengths[i]:.1f}μm, R={dendrite_lengths[resolution-1-i]:.1f}μm"
        )


def polar_bar_plot(
    angles,
    values,
    title="Polar Bar Plot",
    xlabel="Angle (degrees)",
    ylabel="Value",
    threshold=-20.0,
):
    """Create a polar bar plot.

    Args:
        angles (list): List of angles in degrees
        values (list): List of values corresponding to the angles
        title (str): Title of the plot
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
    """
    # when a threshold is crossed, plot this value in a different color
    offset = -1 * np.min(values)  # add to avoid negative bars
    values = [v + offset for v in values]  # Adjust values to avoid negative bars
    angles_rad = np.radians(angles)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    bars = ax.bar(
        angles_rad,
        values,
        width=np.deg2rad(1),
        alpha=0.5,
        bottom=0.0,
        color=["red" if v - offset > threshold else "blue" for v in values],
    )

    ax.set_theta_zero_location("N")  # Optional: 0° at the top
    ax.set_theta_direction(-1)  # Optional: clockwise

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def polar_bar_plot_multi(
    data_dict,
    title="Polar Bar Plot",
    xlabel="Angle (degrees)", 
    ylabel="Value",
    threshold=-20.0,
    colors=None
):
    """Create a polar bar plot with multiple data series.
    
    Args:
        data_dict (dict): Dictionary with {label: (angles, values)} pairs
        title (str): Title of the plot
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        threshold (float): Threshold for color coding
        colors (list): List of colors for each series (optional)
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    # Calculate global offset to avoid negative bars
    all_values = [v for angles, values in data_dict.values() for v in values]
    offset = -1 * np.min(all_values) if np.min(all_values) < 0 else 0
    
    # Calculate bar width based on number of series and angular resolution
    n_series = len(data_dict)
    base_width = np.deg2rad(360 / len(list(data_dict.values())[0][0]))  # Assuming same angles for all
    bar_width = base_width / n_series
    
    for i, (label, (angles, values)) in enumerate(data_dict.items()):
        angles_rad = np.radians(angles)
        adjusted_values = [v + offset for v in values]
        
        # Offset bars slightly for each series
        offset_angles = angles_rad + (i - n_series/2) * bar_width
        
        bars = ax.bar(
            offset_angles,
            adjusted_values,
            width=bar_width,
            alpha=0.7,
            label=label,
            color=colors[i],
            bottom=0.0
        )
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.show()


def polar_bar_plot_grid(
    data_dict,
    title="Polar Bar Plots Grid",
    threshold=-20.0,
    cols=3
):
    """Create a grid of polar bar plots.
    
    Args:
        data_dict (dict): Dictionary with {label: (angles, values)} pairs
        title (str): Overall title
        threshold (float): Threshold for color coding
        cols (int): Number of columns in the grid
    """
    n_plots = len(data_dict)
    rows = (n_plots + cols - 1) // cols  # Ceiling division
    
    fig = plt.figure(figsize=(5*cols, 4*rows))
    fig.suptitle(title, fontsize=16)
    
    # Calculate global offset
    all_values = [v for angles, values in data_dict.values() for v in values]
    offset = -1 * np.min(all_values) if np.min(all_values) < 0 else 0
    
    for i, (label, (angles, values)) in enumerate(data_dict.items()):
        ax = fig.add_subplot(rows, cols, i+1, projection='polar')
        
        angles_rad = np.radians(angles)
        adjusted_values = [v + offset for v in values]
        
        bars = ax.bar(
            angles_rad,
            adjusted_values,
            width=np.deg2rad(1),
            alpha=0.5,
            bottom=0.0,
            color=["red" if v - offset > threshold else "blue" for v in adjusted_values]
        )
        
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Neuron {label}", pad=20)
    
    plt.tight_layout()
    plt.show()


# -- Storage and loading functions --
def store_response_per_angle(
    left_index, angles, all_voltages, max_voltages, spike_counts, filepath=None
):
    """Store the response data in a JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "response_data.json"

    # Load existing data if file exists
    if Path(filepath).exists():
        with open(filepath, "r") as f:
            try:
                response_data = json.load(f)
            except json.JSONDecodeError:
                response_data = {}
    else:
        response_data = {}

    neuron_response = {}
    for i, angle in enumerate(angles):
        neuron_response[str(angle)] = {
            "max_voltage": max_voltages[i],
            "spike_count": int(spike_counts[i]),
            "all_voltages": all_voltages[i].tolist(),
        }

    response_data[str(left_index)] = neuron_response

    with open(filepath, "w") as f:
        json.dump(response_data, f, indent=4)

    print(f"Response data for left_idx {left_index} stored in {filepath}")


def load_response_per_angle(
    left_index, response_filepath=None, min_angle=1, max_angle=360, step=1
):
    """Load response data from a JSON file."""
    if not Path(response_filepath).exists():
        print(f"Response data file {response_filepath} does not exist.")
        return
    with open(response_filepath, "r") as f:
        angles, all_voltages, max_voltages, spike_counts = [], [], [], []
        try:
            response_data = json.load(f)
            if str(left_index) not in response_data:
                print(f)
                print(f"No data for left index {left_index} in response file.")
                return
            response_data = response_data[str(left_index)]
            for angle in range(min_angle, max_angle, step):
                if str(angle) in response_data:
                    data = response_data[str(angle)]
                    angles.append(angle)
                    all_voltages.append(np.array(data["all_voltages"]))
                    max_voltages.append(data["max_voltage"])
                    spike_counts.append(data["spike_count"])
                else:
                    print(f"No data for angle {angle}° in response file.")
        except json.JSONDecodeError:
            print(
                "Response data file is empty or invalid. Please run the simulation first."
            )
            return
    return angles, all_voltages, max_voltages, spike_counts


def load_thresholds(filepath, l=None):
    """Load thresholds from a JSON file."""
    if not Path(filepath).exists:
        print(f"Thresholds file {filepath} does not exist.")
        return {}

    with open(filepath, "r") as f:
        try:
            thresholds = json.load(f)
            threshold = thresholds.get(str(l), None) if l is not None else thresholds
        except json.JSONDecodeError:
            thresholds = {}
            print("JSON file is empty or invalid. Starting fresh.")

    return threshold


if __name__ == "__main__":
    # test
    angle = 90
    arrival_times = calculate_arrival_times(angle)
    print(
        f"Arrival times: Left Ear = {arrival_times[0]:.6f}ms, Right Ear = {arrival_times[1]:.6f}ms"
    )
