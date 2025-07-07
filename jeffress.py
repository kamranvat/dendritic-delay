# simple brian2 implementation of the sound localization model proposed by Jeffress (1948).

from brian2 import *
from utils import *

prefs.codegen.target = (
    "numpy"  # Use numpy instead of Cython to avoid compilation warnings
)


def create_morphologies(resolution=10, max_dendrite_len=100):
    # TODO resolution -> soma
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


def connect_inputs_to_neurons(neurons, input_left, input_right, synapse_weight=2 * mV):
    # TODO move to utils.py
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


def create_jeffress_network(
    resolution=10, max_dendrite_len=100, input_left=None, input_right=None
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
        plot_jeffress_results(
            network["monitors"], network["dendrite_lengths"], network["resolution"]
        )

    if verbose:
        print_jeffress_simulation_info(
            network["dendrite_lengths"], network["resolution"]
        )

    return net


def main():
    start_scope()

    # Sim params
    dt = 0.1 * ms  # Set simulation time step
    sound_frequency = 500  # Hz
    spike_frequency = 350  # Hz
    sim_duration_ms = 50
    sound_angle = 90  # Angle of sound source in degrees

    # TODO set cable properties here, check that all params are set here and shared correctly
    # thin dendrites: 2 um diameter, thick: 4 um diameter

    defaultclock.dt = dt

    # Get phase shift
    phase_left, phase_right = calculate_arrival_times(sound_angle)

    # TODO test jeffress model spike gen: spike_freq = sound_frequency
    # Get input spike times
    left_spike_indices, left_spike_times = binomial_spike_train(
        N=1,
        f_stim_Hz=sound_frequency,
        f_pre_Hz=spike_frequency,
        tmax_ms=sim_duration_ms,
        phase=phase_left,
    )
    right_spike_indices, right_spike_times = binomial_spike_train(
        N=1,
        f_stim_Hz=sound_frequency,
        f_pre_Hz=spike_frequency,
        tmax_ms=sim_duration_ms,
        phase=phase_right,
    )

    # Create input groups
    input_left = SpikeGeneratorGroup(
        1, left_spike_indices, np.array(left_spike_times) * ms
    )
    input_right = SpikeGeneratorGroup(
        1, right_spike_indices, np.array(right_spike_times) * ms
    )

    # Create and simulate network
    network = create_jeffress_network(
        resolution=4,
        max_dendrite_len=80,
        input_left=input_left,
        input_right=input_right,
    )

    simulate_jeffress_network(network, duration=180 * ms, plot=True, verbose=True)


if __name__ == "__main__":
    main()
