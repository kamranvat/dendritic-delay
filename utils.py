import math
import numpy as np


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


# Utility functions for generating spike distributions
def generate_poisson_spikes(rate, duration, n_trials=1, seed=None):
    """Generate Poisson spike trains.

    Args:
        rate (float): Firing rate in Hz
        duration (float): Duration in ms
        n_trials (int): Number of spike trains
        seed (int): Random seed for reproducibility

    Returns:
        list: List of spike times in ms
    """
    if seed is not None:
        np.random.seed(seed)

    spike_times = []
    for trial in range(n_trials):
        # Generate inter-spike intervals from exponential distribution
        intervals = np.random.exponential(
            1000.0 / rate, size=int(rate * duration / 1000 * 3)
        )  # overestimate
        times = np.cumsum(intervals)
        times = times[times < duration]  # keep only spikes within duration
        spike_times.extend(times)

    return sorted(spike_times)


def generate_regular_spikes(interval, duration, start_time=0):
    """Generate regular spike train.

    Args:
        interval (float): Inter-spike interval in ms
        duration (float): Duration in ms
        start_time (float): First spike time in ms

    Returns:
        list: List of spike times in ms
    """
    spike_times = []
    t = start_time
    while t < duration:
        spike_times.append(t)
        t += interval
    return spike_times


def generate_delayed_spikes(base_spikes, delay):
    """Generate delayed version of spike pattern.

    Args:
        base_spikes (list): Base spike times in ms
        delay (float): Delay to add in ms

    Returns:
        list: Delayed spike times in ms
    """
    return [t + delay for t in base_spikes]


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


if __name__ == "__main__":
    # test
    angle = 90
    arrival_times = calculate_arrival_times(angle)
    print(
        f"Arrival times: Left Ear = {arrival_times[0]:.6f}ms, Right Ear = {arrival_times[1]:.6f}ms"
    )
