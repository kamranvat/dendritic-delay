import brian2 as b2
from brian2 import start_scope, NeuronGroup, Synapses, StateMonitor, run, ms
from matplotlib.pyplot import plot, xlabel, ylabel, legend
from neurodynex3.cable_equation import passive_cable
from neurodynex3.tools import input_factory
from brian2 import prefs

# Optional: Set code generation target to numpy if Cython is unavailable
prefs.codegen.target = "numpy"

# Optional: Call passive_cable.getting_started() if needed
passive_cable.getting_started()

def trial():
    start_scope()

    eqs = '''
    dv/dt = (I-v)/tau : 1
    I : 1
    tau : second
    '''
    G = NeuronGroup(2, eqs, threshold='v>1', reset='v = 0', method='exact')
    G.I = [2, 0]
    G.tau = [10, 100]*ms

    # Synapses
    S = Synapses(G, G, on_pre='v_post += 0.2')
    S.connect(i=0, j=1)

    # State Monitor
    M = StateMonitor(G, 'v', record=True)

    # Run simulation
    run(100*ms)

    # Plot results
    plot(M.t/ms, M.v[0], label='Neuron 0')
    plot(M.t/ms, M.v[1], label='Neuron 1')
    xlabel('Time (ms)')
    ylabel('v')
    legend()

# Call the trial function
trial()