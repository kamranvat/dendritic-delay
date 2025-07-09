from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from utils import binomial_spike_train, calculate_arrival_times, polar_bar_plot
from scipy.interpolate import make_interp_spline
from numpy import interp

prefs.codegen.target = "numpy"
defaultclock.dt = 0.01*ms

def excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=10, jitter_ms=0, sound_angle=0,
                         n_comp=11, lambda_um=200, left_comp_index=None, right_comp_index=None, plot = False):
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
    neuron.gl = 0.008*siemens/cm**2 #TODO play with this, this influences threshold crossing!
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

    if plot:
        left_label = f"{left_index}L"
        right_label = f"{right_index - n_comp}R"
        print(f"Left index: {left_label}, Right index: {right_label}")

        plt.plot(M.t/ms, M.v[left_index]/mV, label=f'left dend {left_index}')
        plt.plot(M.t/ms, M.v[0]/mV, label='soma')
        plt.plot(M.t/ms, M.v[right_index]/mV, label=f'right dend {right_index}')
        plt.xlabel('Time (ms)')
        plt.ylabel('v (mV)')
        plt.legend()
        plt.title('Stimulus on dendrites')
        plt.show()

        # Space–time map
        voltmap = np.vstack([
            M.v[n_comp+1:][::-1]/mV,        # right dendrite, distal->proximal
            M.v[0][np.newaxis]/mV,          # soma
            M.v[1:n_comp+1]/mV              # left dendrite, prox->distal
        ])
        times = M.t / ms

        plt.figure(figsize=(10, 4))
        plt.imshow(
            voltmap, aspect='auto', cmap='viridis',
            extent=[times[0], times[-1], dend_length/um, -dend_length/um]
        )
        plt.colorbar(label='Voltage (mV)')
        # Get the limits of the y-axis
        y_min, y_max = plt.ylim()

        # Calculate positions at 1/4, 1/2, and 3/4 of the y-axis
        ytick_positions = [y_min + (y_max - y_min) * 0.25, y_min + (y_max - y_min) * 0.5, y_min + (y_max - y_min) * 0.75]

        # Set the y-ticks and labels
        plt.yticks(ytick_positions, ['left dendrite', 'Soma', 'right dendrite'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Position (μm)')
        plt.title('Space–time voltage map')
        plt.show()



    return max_v

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


def different_angles(n_comp=10, lambda_um=200, left_index=1, right_index=None, min_angle=90, max_angle=270, step=1, plot = True):
    max_voltages = []
    # iterate over different sound angles
    for sound_angle in range(min_angle,max_angle, step):
        print(f"Sound angle: {sound_angle}°")
        max_v = excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=sound_angle, n_comp=n_comp, lambda_um=lambda_um, left_comp_index=left_index,
                right_comp_index=right_index)
        max_voltages.append(max_v)

    angles = np.arange(min_angle, max_angle, step) 
    max_voltages = np.array(max_voltages)  
    # Smooth the data using a moving average
    smoothed_voltages = smooth_data(angles, max_voltages, window_size=20)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(range(min_angle, max_angle, step), max_voltages, marker='o')
        plt.plot(angles, smoothed_voltages, color='red', linewidth=2, label='Smooth fit')
        plt.xlabel('Sound Angle (degrees)')
        plt.ylabel('Max Soma Voltage (mV)')
        plt.title('Max Soma Voltage vs Sound Angle')
        plt.grid()
        plt.tight_layout()
        plt.show()

    return max_voltages

def different_frequencies(min_frequency=50, max_frequency=1001, step=10):
    
    max_voltages = []
    # iterate over different sound frequencies
    for sound_frequency in range(min_frequency, max_frequency, step):        
        print(f"Sound frequency: {sound_frequency} Hz")
        max_v = excite_both_dendrites(N=6, f_stim_Hz=sound_frequency, f_pre_Hz=350, tmax_ms=20, jitter_ms=0, sound_angle=0)
        max_voltages.append(max_v)

    

    # plot max voltages for different sound frequencies
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_frequency, max_frequency, 10), max_voltages, marker='o')
    plt.xlabel('Sound Frequency (Hz)')
    plt.ylabel('Max Soma Voltage (mV)')
    plt.title('Max Soma Voltage vs Sound Frequency')
    plt.grid()
    plt.tight_layout()
    plt.show()

    return max_voltages

def do_polar_plot(left_index, right_index, n_comp=10, lambda_um=200, min_angle=90, max_angle=270, step=1):
    angles = []
    max_voltages = []

    print(f"Left dendrite compartment: {left_index}, Right dendrite compartment: {right_index}")

    # Iterate over sound angles
    for angle in range(min_angle, max_angle, step):
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
        title=f'Max Soma Voltage vs Sound Angle (Left: {left_index}, Right: {right_index})',
        xlabel='Sound Angle (degrees)',
        ylabel='Max Soma Voltage (mV)'
    )

    return max_voltages

def plot_multiple_curves(n_comp=11, lambda_um=200, min_angle=90, max_angle=270, step=1):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_comp))  # Generate distinct colors for each curve

    for left_index in range(1, n_comp + 1):
        right_index = 2 * n_comp + 1 - left_index  # Calculate corresponding right index
        left_label = f"{left_index}L"
        right_label = f"{right_index - n_comp}R"
        print(f"Left index: {left_label}, Right index: {right_label}")

        # Get max voltages for the current combination
        max_voltages = different_angles(
            n_comp=n_comp, lambda_um=lambda_um, left_index=left_index, right_index=right_index,
            min_angle=min_angle, max_angle=max_angle, step=step, plot=False
        )

        angles = np.arange(min_angle, max_angle, step) 
        max_voltages = np.array(max_voltages)  
        # Smooth the data using a moving average
        smoothed_voltages = smooth_data(angles, max_voltages, window_size=20)

        # Only plot the smooth line, no markers!
        plt.plot(
            angles, smoothed_voltages, color=colors[left_index - 1],
            label=f'{left_label}, {right_label}'
        )

    plt.xlabel('Sound Angle (degrees)')
    plt.ylabel('Max Soma Voltage (mV)')
    plt.title('Interpolated Average Curves for Different Indices')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()



# TODO: try input only at one location
def main():

    n_comp = 11
    lambda_um = 200  # from paper
    left_index= 8
    right_index= 2 * n_comp + 1 - left_index
    max_index = n_comp+1

    excite_both_dendrites(N=6, f_stim_Hz=500, f_pre_Hz=350, tmax_ms=10, jitter_ms=0, sound_angle=0, n_comp=n_comp, lambda_um=lambda_um, plot =True, left_comp_index=left_index, right_comp_index=right_index)

    #max_voltages = different_angles(n_comp=n_comp, lambda_um=lambda_um, left_index=left_index, right_index=right_index, min_angle=90, max_angle=270, step=1)

    #max_voltages = different_frequencies(min_frequency=50, max_frequency=1001, step=10)
    
    #for l in range(1, 2):
    #    left_index = l
    #    right_index = 2 * n_comp + 1 - l

    #    max_voltages = do_polar_plot(n_comp=n_comp, lambda_um=lambda_um, min_angle=0, max_angle=360, left_index=left_index, right_index=right_index)
    
    #plot_multiple_curves()


if __name__ == "__main__":
    main()