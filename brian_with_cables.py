from brian2 import *
import matplotlib.pyplot as plt
prefs.codegen.target = "numpy"   # Optional; makes things run without Cython

defaultclock.dt = 0.01*ms

def excite_one_dendrite(dendrite):
    """
    ATTENTION: not up to date, use excite_both_dendrites() instead.
    Excite one of the dendrites (0 for left, 1 for soma, 2 for right).
    dendrite: int, index of the dendrite to excite (0 for left, 1 for soma, 2 for right)
    """

    start_scope()

    # Morphology: L (left dendrite) - soma - R (right dendrite)
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=50*um, diameter=3*um, n=1)
    morpho.R = Cylinder(length=50*um, diameter=3*um, n=1)

    eqs = '''
    Im = gl * (El - v) : amp/meter**2
    gl : siemens/meter**2
    El : volt
    '''

    neuron = SpatialNeuron(
        morphology=morpho,
        model=eqs,
        Cm=1*uF/cm**2,
        Ri=100*ohm*cm,
        method='exponential_euler',
    )

    neuron.v = -65*mV
    neuron.gl = 0.0003*siemens/cm**2
    neuron.El = -65*mV

    # Input group: spike train at given times
    spiketimes = [50, 75, 100, 200]*ms
    input = SpikeGeneratorGroup(1, [0]*len(spiketimes), spiketimes)

    # Connect input to the chosen dendrite
    syn = Synapses(input, neuron, on_pre='v_post += 2*mV')
    syn.connect(i=0, j=dendrite)

    M = StateMonitor(neuron, 'v', record=True)
    run(500*ms)

    plt.plot(M.t/ms, M.v[0]/mV, label='dendL')
    plt.plot(M.t/ms, M.v[1]/mV, label='soma')
    plt.plot(M.t/ms, M.v[2]/mV, label='dendR')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.legend()
    plt.title(f'Stimulus on dendrite index {dendrite}')
    plt.show()

    locations = neuron.distance / um  # or .x for "x coordinate"
    plt.figure(figsize=(10, 4))
    plt.imshow(M.v / mV, aspect='auto',
               extent=[M.t[0] / ms, M.t[-1] / ms, locations[-1], locations[0]],
               cmap="viridis")
    plt.colorbar(label='Membrane potential (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Distance from soma ($\mu$m)')
    plt.title('Space–time voltage map')
    plt.show()



def excite_both_dendrites(time1 = [10,11,12], time2 = [13,14,15]):
    start_scope()

    # Morphology: L (left dendrite) - soma - R (right dendrite)
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=50*um, diameter=10*um, n=1)
    morpho.R = Cylinder(length=50*um, diameter=10*um, n=1)

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

    # Input group: spike train at given times
    spiketimes1 = time1*ms
    input1 = SpikeGeneratorGroup(1, [0]*len(spiketimes1), spiketimes1)

    spiketimes2 = time2*ms
    input2 = SpikeGeneratorGroup(1, [0]*len(spiketimes2), spiketimes2)

    # Connect input to the chosen dendrite
    syn1 = Synapses(input1, neuron, on_pre='v_post += 3*mV')
    syn1.connect(i=0, j=0)

    syn2 = Synapses(input2, neuron, on_pre='v_post += 3*mV')
    syn2.connect(i=0, j=2)

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
    excite_both_dendrites(time1=[5.1,5.2,5.3,5.4,5.5,5.6,5.7], time2=[6.1,6.2,6.3,6.4,6.5,6.6,6.7])

    # Excite only the second dendrite
    #excite_one_dendrite(2)

if __name__ == "__main__":
    main()