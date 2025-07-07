from brian2 import *
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"   # Optional; makes things run without Cython

defaultclock.dt = 0.01*ms

def excite_both_dendrites(timel = [10,11,12], timer = [13,14,15]):
    start_scope()

    # Morphology: L (left dendrite) - soma - R (right dendrite)
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=50*um, diameter=5*um, n=1)
    morpho.R = Cylinder(length=50*um, diameter=5*um, n=1)

    eqs = '''
    Im = gl * (El - v) : amp/meter**2
    gl : siemens/meter**2
    El : volt
    '''

    neuron = SpatialNeuron(
        morphology=morpho,
        model=eqs,
        threshold='v > -55*mV',
        threshold_location=1,  # Soma
        reset='v = -65*mV',
        refractory='2*ms',
        Cm=1*uF/cm**2,
        Ri=100*ohm*cm,   # (default/recommended)
        method='exponential_euler',
    )

    neuron.v = -65*mV
    neuron.gl = 0.00005*siemens/cm**2  # tau=20 ms
    neuron.El = -65*mV

    # Generate 6 spike times for 6 axons arriving at the same time (say, t=10 ms)
    left_spiketimes = timel*ms
    right_spiketimes = timer*ms

    #indices are lists each containing n=timel and n=timer axons
    left_indices = list(range(len(timel)))  
    right_indices = list(range(len(timer)))  

    # Create input groups (6 axons each)
    input_left = SpikeGeneratorGroup(len(timel), left_indices, left_spiketimes)
    input_right = SpikeGeneratorGroup(len(timer), right_indices, right_spiketimes)

    # Synapses: connect ALL left inputs to dendrite 0; right inputs to dendrite 2
    syn_left = Synapses(input_left, neuron, on_pre='v_post += 1*mV')
    syn_left.connect(i=range(len(timel)), j=0)  # all to compartment 0

    syn_right = Synapses(input_right, neuron, on_pre='v_post += 1*mV')
    syn_right.connect(i=range(len(timer)), j=2)  # all to compartment 2


    M = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)
    run(20*ms)

    print("Max soma voltage:", np.max(M.v[1]/mV), "mV")

    if spikemon.count[1] > 0:
        print("Soma spiked!")
        print("Spike times (ms):", spikemon.t/ms)
    else:
        print("Soma did NOT spike.")

    plt.plot(M.t/ms, M.v[0]/mV, label='dendL')
    plt.plot(M.t/ms, M.v[1]/mV, label='soma')
    plt.plot(M.t/ms, M.v[2]/mV, label='dendR')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.legend()
    plt.title(f'Stimulus on dendrites')
    plt.show()

    locations = neuron.distance / um  
    plt.figure(figsize=(10, 4))
    plt.imshow(M.v / mV, aspect='auto',
               extent=[M.t[0] / ms, M.t[-1] / ms, locations[-1], locations[0]],
               cmap="viridis")
    plt.colorbar(label='Membrane potential (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Distance from soma ($\mu$m)')
    plt.title('Space–time voltage map')
    plt.show()

def main():
    # Excite the first dendrite
    #excite_both_dendrites(timel=[5.1,5.2,5.3,5.4,5.5,5.6], timer=[6.1,6.2,6.3,6.4,6.5,6.6])
    excite_both_dendrites(timel=[5,5,5,5,5,5], timer=[6,6,6,6,6,6])

    # Excite only the second dendrite
    #excite_one_dendrite(2)

if __name__ == "__main__":
    main()