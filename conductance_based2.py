from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from utils import binomial_spike_train, calculate_arrival_times

prefs.codegen.target = "numpy"
defaultclock.dt = 0.01*ms

def excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=0, dend_length=40):
    start_scope()

    # Morphology
    dend_length = dend_length*um  # convert to meters
    diameter = 2*um
    lambda_ = 200 * um 
    

    compartment_length = 0.05 * lambda_ 
    n_comp = int(np.round(dend_length / compartment_length))

    morpho = Soma(diameter=20*um)
    morpho.L = Cylinder(length=dend_length, diameter=diameter, n=n_comp)
    morpho.R = Cylinder(length=dend_length, diameter=diameter, n=n_comp)

    #length = lengthright*um  # or lengthleft, lengthright depending which dendrite
    #area = np.pi * diameter * length   # ~251 um^2

    #w_e_total = 24*nS   # 22 nS per whole synapse event (as in the paper)
    #w_syn = w_e_total / area
    

    eqs = '''
    Im = gl * (El - v) + gsyn*(Esyn-v) : amp/meter**2
    dgsyn/dt = -gsyn / tau_syn : siemens/meter**2
    gl : siemens/meter**2
    El : volt
    Esyn : volt
    tau_syn : second (shared)
    '''

    neuron = SpatialNeuron(
        morphology=morpho,
        model=eqs,
        threshold='v > -55*mV',
        threshold_location=0,
        reset='v = -65*mV',
        refractory='2*ms',
        Cm=1*uF/cm**2,
        Ri=200*ohm*cm,
        method='exponential_euler',
    )

    neuron.v = -65*mV
    #neuron.gl = 0.001*siemens/cm**2  # τ = 1 ms (typical)
    neuron.gl = 0.0005*siemens/cm**2 # τ = 2 ms (like paper)
    neuron.El = -62.5*mV
    neuron.gsyn = 0*siemens/cm**2
    neuron.Esyn = 0*mV   
    neuron.tau_syn = 0.5*ms  # synaptic time constant (as in the paper)
    w_syn = 14*nS # 14 to 26


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

    # calculate compartment where synapses are located
    # List all compartment edges:
    L = dend_length    # total dendrite length in um
    n_comp = int(np.round(L / compartment_length))
    compartment_centers = np.linspace(compartment_length/2, L - compartment_length/2, n_comp)  # [5, 15, 25, 35] um for 4 comps
    # Find index of compartment closest to 0.1*lambda_
    syn_dist = 0.1 * lambda_   # 20 μm
    left_index = 1 + np.argmin(np.abs(compartment_centers - syn_dist))
    right_index = n_comp + 1 + np.argmin(np.abs(compartment_centers - syn_dist))

    syn_left = Synapses(input_left, neuron, on_pre='gsyn_post += w_syn / area_post')
    syn_left.connect(i=range(N), j=left_index)

    syn_right = Synapses(input_right, neuron, on_pre='gsyn_post += w_syn / area_post')
    syn_right.connect(i=range(N), j=right_index)

    M = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)
    run(tmax_ms*ms)

    print("Max soma voltage:", np.max(M.v[0]/mV), "mV")
    if spikemon.count[0] > 0:
        print("Soma spiked!")
        print("Spike times (ms):", spikemon.t/ms)
    else:
        print("Soma did NOT spike.")

    # Python

    plt.plot(M.t/ms,M.v[left_index]/mV, label='dendL')  # Plot mean voltage
    plt.plot(M.t/ms, M.v[0]/mV, label='soma')
    plt.plot(M.t/ms, M.v[right_index]/mV, label='dendR')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.legend()
    plt.title('Stimulus on dendrites')
    plt.show()

    # Stack voltage traces in dendrite-soma-dendrite (vertical) order
    voltmap = np.vstack([
        M.v[n_comp+1:][::-1]/mV,   # right dendrite, distal->proximal
        M.v[0][np.newaxis]/mV,     # soma
        M.v[1:n_comp+1]/mV         # left dendrite, proximal->distal
    ])
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
    plt.yticks([left_index, 0, right_index], ['right dendrite', 'Soma', 'left dendrite'])
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