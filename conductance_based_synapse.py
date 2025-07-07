from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from utils import binomial_spike_train, calculate_arrival_times

prefs.codegen.target = "numpy"
defaultclock.dt = 0.01*ms

def excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=0, lengthleft=58, lengthright=58):
    start_scope()

    # Morphology
    diameter = 4*um
    morpho = Soma(diameter=20*um)
    morpho.L = Cylinder(length=lengthleft*um, diameter=diameter, n=1)
    morpho.R = Cylinder(length=lengthright*um, diameter=diameter, n=1)

    length = lengthright*um  # or lengthleft, lengthright depending which dendrite
    area = np.pi * diameter * length   # ~251 um^2

    w_e_total = 24*nS   # 22 nS per whole synapse event (as in the paper)
    w_e = w_e_total / area       # units: S / um^2

    tau_e = 0.3*ms  # synaptic time constant (as in the paper)#

    eqs = '''
    Im = gl * (El - v) + ge * (Ee - v) : amp/meter**2
    dge/dt = -ge/tau_e : siemens/meter**2 
    gl : siemens/meter**2
    El : volt
    Ee : volt
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
    neuron.gl = 0.001*siemens/cm**2  # τ = 1 ms (typical)
    #neuron.gl = 0.0005*siemens/cm**2 # τ = 2 ms (like paper)
    neuron.El = -65*mV
    neuron.Ee = 0*mV

    # calculate arrival times for both dendrites
    time_left, time_right = calculate_arrival_times(sound_angle)
    
    # normalize the times to the cycle length with left as reference
    itd = time_right - time_left

    # Binomial spike trains per dendrite
    left_i, left_t = binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=0, jitter_ms=jitter_ms)
    right_i, right_t = binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=itd, jitter_ms=jitter_ms)

    # 2. Concatenate ALL spike times (could be negative)
    all_times = np.concatenate([left_t, right_t])
    min_time = np.min(all_times)

    if min_time < 0:
        # 3. Shift ALL spike times so the earliest is at 0 ms
        left_t = np.array(left_t) - min_time
        right_t = np.array(right_t) - min_time
        # 4. Also increase tmax to accommodate this shift
        tmax_ms = tmax_ms - min_time  # (since -min_time is positive if min_time is negative)

    input_left = SpikeGeneratorGroup(N, left_i, np.array(left_t)*ms)
    input_right = SpikeGeneratorGroup(N, right_i, np.array(right_t)*ms)

    syn_left = Synapses(input_left, neuron, on_pre='ge_post += w_e')
    syn_left.connect(i=range(N), j=1)

    syn_right = Synapses(input_right, neuron, on_pre='ge_post += w_e')
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
    plt.yticks([lengthright, 0, lengthleft], ['right dendrite', 'Soma', 'left dendrite'])
    plt.xlabel('Time (ms)')
    plt.ylabel('Position (μm)')
    plt.title('Space–time voltage map')
    plt.show()

#TODO: conductance based or current based synapses?
def main():
    excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=0)

    # iterate over different sound angles
    #for sound_angle in range(0, 360, 30):
    #    print(f"Sound angle: {sound_angle}°")
    #    excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=sound_angle)

    # iterate over different sound frequencies
    #for sound_frequency in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    #    print(f"Sound frequency: {sound_frequency} Hz")
    #    excite_both_dendrites(N=6, f_stim_Hz=sound_frequency, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=0)

if __name__ == "__main__":
    main()