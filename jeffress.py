# simple brian2 implementation of the sound localization model proposed by Jeffress (1948).

from brian2 import *
from utils import *

prefs.codegen.target = (
    "numpy"  # Use numpy instead of Cython to avoid compilation warnings
)


def create_morphologies(resolution=10, max_dendrite_len=100):
    """Create morphologies for Jeffress model neurons.

    Args:
        resolution (int): Number of neurons to create
        max_dendrite_len (float): Maximum dendrite length in micrometers

    Returns:
        tuple: (morphologies, dendrite_lengths)
    """
    # get the lengths of the dendrites (linearly spaced)
    lengths = np.linspace(0, max_dendrite_len, resolution + 2)[
        1:-1
    ]  # avoid zero length dendrites

    morphologies = []

    # build each soma with its dendrites
    for i in range(resolution):
        soma = Soma(diameter=30 * um)

        # left dendrite increases, right dendrite decreases linearly
        left_length = lengths[i]
        right_length = lengths[resolution - 1 - i]

        # add dendrites to soma
        soma.left = Cylinder(length=left_length * um, diameter=2 * um, n=1)
        soma.right = Cylinder(length=right_length * um, diameter=2 * um, n=1)

        morphologies.append(soma)

    return morphologies, lengths


def create_neurons(morphologies):
    """Create spatial neurons from morphologies.

    Args:
        morphologies (list): List of Brian2 morphology objects

    Returns:
        list: List of SpatialNeuron objects
    """
    eqs = """
    Im = gl * (El - v) : amp/meter**2
    gl : siemens/meter**2
    El : volt
    """

    neurons = []
    for morph in morphologies:
        neuron = SpatialNeuron(
            morphology=morph,
            model=eqs,
            Cm=1 * uF / cm**2,
            Ri=100 * ohm * cm,
            method="exponential_euler",
        )

        # neuron params
        neuron.v = -65 * mV
        neuron.gl = 0.0003 * siemens / cm**2
        neuron.El = -65 * mV

        neurons.append(neuron)

    return neurons


def create_input_groups(left_spike_times, right_spike_times):
    """Create spike generator groups for left and right inputs.

    Args:
        left_spike_times (array-like): Spike times for left input (in ms)
        right_spike_times (array-like): Spike times for right input (in ms)

    Returns:
        tuple: (input_left, input_right) SpikeGeneratorGroup objects
    """
    # Convert to Brian2 time units if needed
    if hasattr(left_spike_times, "unit"):
        left_times = left_spike_times
    else:
        left_times = np.array(left_spike_times) * ms

    if hasattr(right_spike_times, "unit"):
        right_times = right_spike_times
    else:
        right_times = np.array(right_spike_times) * ms

    input_left = SpikeGeneratorGroup(1, [0] * len(left_times), left_times)
    input_right = SpikeGeneratorGroup(1, [0] * len(right_times), right_times)

    return input_left, input_right


def connect_inputs_to_neurons(neurons, input_left, input_right, synapse_weight=2 * mV):
    """Connect input groups to neurons via synapses.

    Args:
        neurons (list): List of SpatialNeuron objects
        input_left (SpikeGeneratorGroup): Left input group
        input_right (SpikeGeneratorGroup): Right input group
        synapse_weight (Quantity): Synaptic weight (default: 2*mV)

    Returns:
        tuple: (synapses_left, synapses_right) lists of synapse objects
    """
    synapses_left = []
    synapses_right = []

    for i, neuron in enumerate(neurons):
        # left input connects to left dendrite (compartment 0)
        syn_left = Synapses(
            input_left, neuron, on_pre=f"v_post += {synapse_weight/mV}*mV"
        )
        syn_left.connect(i=0, j=0)  # connect to left dendrite
        synapses_left.append(syn_left)

        # right input connects to right dendrite (compartment 2)
        syn_right = Synapses(
            input_right, neuron, on_pre=f"v_post += {synapse_weight/mV}*mV"
        )
        syn_right.connect(i=0, j=2)  # connect to right dendrite
        synapses_right.append(syn_right)

    return synapses_left, synapses_right


def setup_monitors(neurons):
    """Set up state monitors for all neurons.

    Args:
        neurons (list): List of SpatialNeuron objects

    Returns:
        list: List of StateMonitor objects
    """
    monitors = []
    for neuron in neurons:
        monitor = StateMonitor(neuron, "v", record=True)
        monitors.append(monitor)
    return monitors


def run_simulation(neurons, input_groups, synapses, monitors, duration=200 * ms):
    """Run the Brian2 simulation.

    Args:
        neurons (list): List of neurons
        input_groups (tuple): (input_left, input_right)
        synapses (tuple): (synapses_left, synapses_right)
        monitors (list): List of monitors
        duration (Quantity): Simulation duration

    Returns:
        Network: The Brian2 network object after simulation
    """
    input_left, input_right = input_groups
    synapses_left, synapses_right = synapses

    # create network with all objects
    network_objects = (
        [input_left, input_right] + neurons + synapses_left + synapses_right + monitors
    )
    net = Network(*network_objects)

    # run simulation
    net.run(duration)

    return net


def plot_results(
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


def print_simulation_info(dendrite_lengths, resolution):
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


def create_jeffress_network(
    resolution=10, max_dendrite_len=100, left_spike_times=[50], right_spike_times=[75]
):
    """Create a complete Jeffress model network (high-level function).

    Args:
        resolution (int): Number of neurons
        max_dendrite_len (float): Maximum dendrite length in micrometers
        left_spike_times (list): Spike times for left input in ms
        right_spike_times (list): Spike times for right input in ms

    Returns:
        dict: Dictionary containing all network components
    """
    start_scope()

    # Create morphologies and neurons
    morphologies, dendrite_lengths = create_morphologies(resolution, max_dendrite_len)
    neurons = create_neurons(morphologies)

    # Create inputs
    input_left, input_right = create_input_groups(left_spike_times, right_spike_times)

    # Connect inputs to neurons
    synapses_left, synapses_right = connect_inputs_to_neurons(
        neurons, input_left, input_right
    )

    # Setup monitoring
    monitors = setup_monitors(neurons)

    return {
        "neurons": neurons,
        "morphologies": morphologies,
        "dendrite_lengths": dendrite_lengths,
        "inputs": (input_left, input_right),
        "synapses": (synapses_left, synapses_right),
        "monitors": monitors,
        "resolution": resolution,
    }


def simulate_jeffress_network(network, duration=200 * ms, plot=True, verbose=True):
    """Simulate a Jeffress network and optionally plot results.

    Args:
        network (dict): Network dictionary from create_jeffress_network()
        duration (Quantity): Simulation duration
        plot (bool): Whether to plot results
        verbose (bool): Whether to print info

    Returns:
        Network: Brian2 network object after simulation
    """
    net = run_simulation(
        network["neurons"],
        network["inputs"],
        network["synapses"],
        network["monitors"],
        duration,
    )

    if plot:
        plot_results(
            network["monitors"], network["dendrite_lengths"], network["resolution"]
        )

    if verbose:
        print_simulation_info(network["dendrite_lengths"], network["resolution"])

    return net


def main():
    # Example 1: Using the high-level function (easiest)
    print("=== Example 1: High-level function ===")
    network = create_jeffress_network(
        resolution=5,
        max_dendrite_len=100,
        left_spike_times=[30, 60, 90],
        right_spike_times=[45, 75, 105],
    )
    simulate_jeffress_network(network, duration=150 * ms)

    print("\n=== Example 2: Manual step-by-step construction ===")
    # Example 2: Using individual functions for more control
    start_scope()

    # Create morphologies and neurons
    morphologies, dendrite_lengths = create_morphologies(
        resolution=3, max_dendrite_len=50
    )
    neurons = create_neurons(morphologies)

    # Create custom spike patterns (utility function could generate these)
    left_spikes = [20, 40, 80]  # ms
    right_spikes = [25, 50, 85]  # ms

    input_left, input_right = create_input_groups(left_spikes, right_spikes)
    synapses_left, synapses_right = connect_inputs_to_neurons(
        neurons, input_left, input_right, synapse_weight=3 * mV
    )
    monitors = setup_monitors(neurons)

    # Run simulation
    net = run_simulation(
        neurons,
        (input_left, input_right),
        (synapses_left, synapses_right),
        monitors,
        120 * ms,
    )

    # Plot and print results
    plot_results(monitors, dendrite_lengths, 3, "Custom spike pattern simulation")
    print_simulation_info(dendrite_lengths, 3)

    # Example 3: Using utility functions
    example_with_utility_functions()


def example_with_utility_functions():
    """Example showing how to use utility functions with the modular Jeffress model."""
    print("\n=== Example 3: Using utility functions for spike generation ===")

    # Generate different spike patterns
    base_pattern = generate_regular_spikes(interval=30, duration=150, start_time=10)
    left_spikes = base_pattern
    right_spikes = generate_delayed_spikes(base_pattern, delay=5)  # 5ms delay

    print(f"Left spikes: {left_spikes}")
    print(f"Right spikes: {right_spikes}")

    # Create and simulate network
    network = create_jeffress_network(
        resolution=4,
        max_dendrite_len=80,
        left_spike_times=left_spikes,
        right_spike_times=right_spikes,
    )
    simulate_jeffress_network(network, duration=180 * ms)


if __name__ == "__main__":
    main()
