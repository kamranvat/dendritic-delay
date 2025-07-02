import brian2 as b2
from brian2 import start_scope, NeuronGroup, Synapses, StateMonitor, run, ms
import matplotlib.pyplot as plt
from neurodynex3.cable_equation import passive_cable
from neurodynex3.tools import input_factory
from brian2 import prefs

# Optional: Set code generation target to numpy if Cython is unavailable
prefs.codegen.target = "numpy"

# Optional: Call passive_cable.getting_started() if needed
#passive_cable.getting_started()

# Define the network
def create_network():
    start_scope()

    # Neuron model equations
    eqs = '''
    dv/dt = (I - v) / tau : 1
    I : 1  # Input current
    tau : second  # Membrane time constant
    '''

    # Create 3 neurons: 2 input neurons and 1 output neuron
    neurons = NeuronGroup(3, eqs, threshold='v > 0.99', reset='v = 0', method='exact')
    neurons.I = [1.0, 1.0, 0.1]  # Input currents for neurons
    neurons.tau = [10, 10, 100] * ms  # Membrane time constants

    # Synapses: Connect input neurons (0 and 1) to the output neuron (2)
    syn = Synapses(neurons, neurons, on_pre='v_post += 0.5')  # Add 0.5 to output neuron's v
    syn.connect(i=[0, 1], j=[2, 2])  # Connect neuron 0 and 1 to neuron 2

    # State monitor to record membrane potentials
    monitor = StateMonitor(neurons, 'v', record=True)

    # Run the simulation
    run(100 * ms)

    # Plot results
    for i in range(3):
        plt.plot(monitor.t / ms, monitor.v[i], label=f'Neuron {i}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (v)')
    plt.legend()
    plt.show()

# Call the function to create and simulate the network
create_network()