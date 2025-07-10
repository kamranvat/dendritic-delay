
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from utils import binomial_spike_train, calculate_arrival_times, polar_bar_plot
from brian2tools import *

prefs.codegen.target = "numpy"
defaultclock.dt = 0.01*ms

def excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=0,
                         n_comp=10, lambda_um=200, left_comp_index=None, right_comp_index=None):
    start_scope()

    # Key morphology
    lambda_ = lambda_um * um
    compartment_length = 0.05 * lambda_
    dend_length = n_comp * compartment_length
    diameter = 2*um  # From paper

    morpho = Soma(diameter=20*um)
    morpho.L = Cylinder(length=dend_length, diameter=diameter, n=n_comp)
    morpho.R = Cylinder(length=dend_length, diameter=diameter, n=n_comp)

    #show()

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
    #neuron.gl = 0.0005*siemens/cm**2
    neuron.gl = 0.001*siemens/cm**2

    neuron.El = -62.5*mV
    neuron.gsyn = 0*siemens/cm**2
    neuron.Esyn = 0*mV
    neuron.tau_syn = 0.5*ms
    w_syn = 14*nS  # (14-26 in paper)



    # Calculate arrival times for both dendrites
    time_left, time_right = calculate_arrival_times(sound_angle)
    itd = time_right - time_left

    left_i, left_t = binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=0, jitter_ms=jitter_ms)
    right_i, right_t = binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=itd, jitter_ms=jitter_ms)

    # Shift times if negative
    all_times = np.concatenate([left_t, right_t])
    min_time = np.min(all_times)
    if min_time < 0:
        left_t, right_t = left_t - min_time, right_t - min_time
        tmax_ms = tmax_ms - min_time

    input_left = SpikeGeneratorGroup(N, left_i, left_t*ms)
    input_right = SpikeGeneratorGroup(N, right_i, right_t*ms)

    # Compartment centers (for info/indices)
    compartment_centers = np.linspace(compartment_length/2, dend_length - compartment_length/2, n_comp)
    syn_dist = 0.1 * lambda_   # 0.1 lambda as in paper

    # Defaults: closest to 0.1 lambda in each dendrite
    default_left = 1 + np.argmin(np.abs(compartment_centers - syn_dist))
    default_right = n_comp + 1 + np.argmin(np.abs(compartment_centers - syn_dist))
    left_index = default_left if left_comp_index is None else left_comp_index
    right_index = default_right if right_comp_index is None else right_comp_index

    # Synapse connections
    syn_left = Synapses(input_left, neuron, on_pre='gsyn_post += w_syn / area_post')
    syn_left.connect(i=range(N), j=left_index)
    syn_right = Synapses(input_right, neuron, on_pre='gsyn_post += w_syn / area_post')
    syn_right.connect(i=range(N), j=right_index)

    M = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)
    run(tmax_ms*ms)

    print(f"[INFO] n_comp={n_comp}, lambda={lambda_um} um, dendrite length={dend_length/um:.1f} um, left_index={left_index}, right_index={right_index}")
    max_v = np.max(M.v[0]/mV)
    print("Max soma voltage:", max_v, "mV")
    if spikemon.count[0] > 0:
        print("Soma spiked!")
        print("Spike times (ms):", spikemon.t/ms)
    else:
        print("Soma did NOT spike.")

    """plt.plot(M.t/ms, M.v[left_index]/mV, label=f'left dend {left_index}')
    plt.plot(M.t/ms, M.v[0]/mV, label='soma')
    plt.plot(M.t/ms, M.v[right_index]/mV, label=f'right dend {right_index}')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.legend()
    plt.title('Stimulus on dendrites')
    #plt.show()"""

    # Space–time map
    voltmap = np.vstack([
        M.v[n_comp+1:][::-1]/mV,        # right dendrite, distal->proximal
        M.v[0][np.newaxis]/mV,          # soma
        M.v[1:n_comp+1]/mV              # left dendrite, prox->distal
    ])
    times = M.t / ms

    """plt.figure(figsize=(10, 4))
    plt.imshow(
        voltmap, aspect='auto', cmap='viridis',
        extent=[times[0], times[-1], dend_length/um, -dend_length/um]
    )
    plt.colorbar(label='Voltage (mV)')
    plt.yticks([left_index, 0, right_index], ['right dendrite', 'Soma', 'left dendrite'])
    plt.xlabel('Time (ms)')
    plt.ylabel('Position (μm)')
    plt.title('Space–time voltage map')
    #plt.show()"""

    return max_v


def plot_dendrite_structure(n_comp=10, lambda_um=200, show_labels=True):
    compartment_length = 0.05 * lambda_um
    dend_length = n_comp * compartment_length

    # Soma in the middle (x=0, y=0) as a big circle
    soma_x, soma_y = [0], [0]

    # Left dendrite: list of points from soma to distal tip (negative x)
    left_x = np.linspace(0, -dend_length, n_comp+1)
    left_y = np.zeros_like(left_x)
    # Right dendrite: list of points from soma to distal tip (positive x)
    right_x = np.linspace(0, dend_length, n_comp+1)
    right_y = np.zeros_like(right_x)

    plt.figure(figsize=(8, 2))
    # Plot soma
    plt.plot(soma_x, soma_y, 'ko', markersize=16, label='Soma (comp 0)')
    # Plot dendrites as connected lines
    plt.plot(left_x, left_y, color='b', linewidth=2, label='Left dendrite')
    plt.plot(right_x, right_y, color='r', linewidth=2, label='Right dendrite')

    # Mark compartment centers
    # Left dendrite: comps 1..n_comp, centers NOT including soma
    for i in range(1, n_comp+1):
        x = -compartment_length * i + compartment_length/2
        plt.plot(x, 0, 'bs')
        if show_labels:
            plt.text(x, 0.1, f'{i}', color='b', ha='center', fontsize=9)

    # Right dendrite: comps n_comp+1..2*n_comp
    for i in range(1, n_comp+1):
        x = compartment_length * i - compartment_length/2
        plt.plot(x, 0, 'rs')
        if show_labels:
            plt.text(x, -0.1, f'{n_comp + i}', color='r', ha='center', fontsize=9)

    plt.text(0, 0.2, "0", color='k', ha='center', fontsize=10)  # Soma index

    plt.xlabel('Distance (μm)')
    plt.ylabel('')
    plt.yticks([])
    plt.title("Dendritic Morphology and Compartment Indices")
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    n_comp = 10
    lambda_um = 200  # from paper

    #plot_dendrite_structure(n_comp=n_comp, lambda_um=lambda_um, show_labels=True)
    max_voltages = []
    angles = []

    #for l in range(1, n_comp+1):
    for l in range(1, 2):
        # Loop: l = 1..n_comp, r = n_comp..1
        left_index = 1
        right_index = 2 * n_comp + 1 - l
        print(f"Left dendrite compartment: {left_index}, Right dendrite compartment: {right_index}")

        for angle in range(90, 270, 1):
            print(f"Sound angle: {angle}°")
                
            max_v = excite_both_dendrites(
                N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=angle,
                n_comp=n_comp, lambda_um=lambda_um,
                left_comp_index=left_index,
                right_comp_index=right_index
            )
            max_voltages.append(max_v)
            angles.append(angle)

        # Plot max voltages for different sound angles
        polar_bar_plot(
            angles, max_voltages,
            title='Max Soma Voltage vs Sound Angle',
            xlabel='Sound Angle (degrees)',
            ylabel='Max Soma Voltage (mV)'
        )
            

if __name__ == "__main__":
    main()