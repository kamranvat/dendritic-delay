from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from utils import binomial_spike_train

prefs.codegen.target = "numpy"
defaultclock.dt = 0.01*ms

def excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0):
    start_scope()

    # Morphology
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=50*um, diameter=2*um, n=1)
    morpho.R = Cylinder(length=50*um, diameter=2*um, n=1)

    eqs = '''
    Im = gl * (El - v) : amp/meter**2
    gl : siemens/meter**2
    El : volt
    '''

    neuron = SpatialNeuron(
        morphology=morpho,
        model=eqs,
        threshold='v > -55*mV',
        threshold_location=0,
        reset='v = -65*mV',
        refractory='2*ms',
        Cm=1*uF/cm**2,
        Ri=100*ohm*cm,
        method='exponential_euler',
    )

    neuron.v = -65*mV
    neuron.gl = 0.00005*siemens/cm**2
    neuron.El = -65*mV

    # Binomial spike trains per dendrite
    left_i, left_t = binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=0, jitter_ms=jitter_ms)
    right_i, right_t = binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=1, jitter_ms=jitter_ms)

    input_left = SpikeGeneratorGroup(N, left_i, np.array(left_t)*ms)
    input_right = SpikeGeneratorGroup(N, right_i, np.array(right_t)*ms)

    syn_left = Synapses(input_left, neuron, on_pre='v_post += 3*mV')
    syn_left.connect(i=range(N), j=1)

    syn_right = Synapses(input_right, neuron, on_pre='v_post += 3*mV')
    syn_right.connect(i=range(N), j=2)

    M = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)
    run(tmax_ms*ms)

    print("Max soma voltage:", np.max(M.v[1]/mV), "mV")
    if spikemon.count[0] > 0:
        print("Soma spiked!")
        print("Spike times (ms):", spikemon.t/ms)
    else:
        print("Soma did NOT spike.")

    plt.plot(M.t/ms, M.v[1]/mV, label='dendL')
    plt.plot(M.t/ms, M.v[0]/mV, label='soma')
    plt.plot(M.t/ms, M.v[2]/mV, label='dendR')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.legend()
    plt.title('Stimulus on dendrites')
    plt.show()

    # Stack voltage traces in dendrite-soma-dendrite (vertical) order
    voltmap = np.vstack([M.v[1]/mV, M.v[0]/mV, M.v[2]/mV])
    times = M.t / ms

    plt.figure(figsize=(10, 4))
    plt.imshow(
        voltmap,
        aspect='auto',
        cmap='viridis',
        # Y runs top (right) to bottom (left), spanning the actual distance
        extent=[times[0], times[-1], 50, -50]
    )
    plt.colorbar(label='Voltage (mV)')
    plt.yticks([50, 0, -50], ['right dend', 'soma', 'left dend'])
    plt.xlabel('Time (ms)')
    plt.ylabel('Position (μm)')
    plt.title('Space–time voltage map')
    plt.show()


#TODO: 
# 1. use kamrans winkel to calculate the delay
# 2. use values from the paper for phase = 0, frequency = ca. 500Hz, jitter = 0
# frequency gegen response plotten
# winkel gegen response plotten
def main():
    excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0)

if __name__ == "__main__":
    main()