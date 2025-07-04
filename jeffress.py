# simple brian2 implementation of the sound localization model proposed by Jeffress (1948).

from brian2 import *
#prefs.codegen.target = "numpy"   # Optional; makes things run without Cython

def excite_one_dendrite(dendrite):
    # demo function: excite one of the dendrites of a neuron, plot the results
    start_scope()

    # Morphology: L (left dendrite) - soma - R (right dendrite)
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=100*um, diameter=2*um, n=1)
    morpho.R = Cylinder(length=100*um, diameter=2*um, n=1)

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

    M = StateMonitor(neuron, 'v', record=[0, 1, 2])
    run(500*ms)

    plot(M.t/ms, M.v[0]/mV, label='dendL')
    plot(M.t/ms, M.v[1]/mV, label='soma')
    plot(M.t/ms, M.v[2]/mV, label='dendR')
    xlabel('Time (ms)')
    ylabel('v (mV)')
    legend()
    title(f'Stimulus on dendrite index {dendrite}')
    show()


def excite_both_dendrites(time1 = [10,11,12], time2 = [13,14,15]):
    start_scope()

    # Morphology: L (left dendrite) - soma - R (right dendrite)
    morpho = Soma(diameter=30*um)
    morpho.L = Cylinder(length=100*um, diameter=2*um, n=1)
    morpho.R = Cylinder(length=100*um, diameter=2*um, n=1)

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
    syn1 = Synapses(input1, neuron, on_pre='v_post += 1*mV')
    syn1.connect(i=0, j=0)

    syn2 = Synapses(input2, neuron, on_pre='v_post += 1*mV')
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



def calculate_morphology(resolution=10, secondary_stem_len=1, secondary_branch_len=100):
    # calculate a morphology with a given resolution and delay line length
    # setup: two somas with one delay line each, splitting into N_resulution dendrites
    # the N_resolution dendrites run towards each other, with a synapse in the middle

    # define "ears"
    ear_L = Soma(diameter=30*um)
    ear_R = Soma(diameter=30*um)

    # add one wire to each soma (secondary stem, ie the delay line)
    secondary_stem = Cylinder(length=secondary_stem_len*um, diameter=2*um, n=1)

    ear_L.stem = secondary_stem
    ear_R.stem = secondary_stem

    # get the lengths of the dendrites after they branch out
    lengths = np.linspace(0, secondary_branch_len, resolution)
    branches = [Cylinder(length=l*um, diameter=2*um, n=1) for l in lengths]

    # define synapses
    w = 1*mV 
    S = Synapses(ear_L, ear_R, on_pre='v_post += w') # simple readout

    # add the dendrites to the stems (reversed order for one of them)
    for i, branch in enumerate(branches):
        branch_name = f'branch_{i}'

        ear_L["stem"][branch_name] = branch
        ear_R["stem"][branch_name] = (branches[resolution - 1 - i])

        # define tertiary fibers as synapses between the corresponding branch dendrites
        # S.connect(i=ear_L["stem"][branch_name], j=ear_R["stem"][branch_name])

    eqs = '''
    Im = gl * (El - v) : amp/meter**2
    gl : siemens/meter**2
    El : volt
    '''

    # build the neurons:
    neuron_L = SpatialNeuron(
        morphology=ear_L,
        model=eqs,
        Cm=1*uF/cm**2,
        Ri=100*ohm*cm,
        method='exponential_euler',
    )
    neuron_R = SpatialNeuron(
        morphology=ear_R,
        model=eqs,
        Cm=1*uF/cm**2,
        Ri=100*ohm*cm,
        method='exponential_euler',
    )

    neuron_L.v, neuron_R.v = -65*mV, -65*mV
    neuron_L.gl, neuron_R.gl = 0.0003*siemens/cm**2, 0.0003*siemens/cm**2
    neuron_L.El, neuron_R.El = -65*mV, -65*mV

    # give each soma an input synapse
    input = SpikeGeneratorGroup(1, [0]*resolution, np.arange(0, resolution)*100*ms)
    syn_input_L = Synapses(input, neuron_L, on_pre='v_post += 2*mV')
    syn_input_R = Synapses(input, neuron_R, on_pre='v_post += 2*mV')
    syn_input_L.connect(i=0, j=0)  # connect input to left neuron
    syn_input_R.connect(i=0, j=0)  # connect input to right neuron

    # connect the neurons with the synapses
    for i in range(resolution):
        S.connect(i=ear_L["stem"][f'branch_{i}'], j=ear_R["stem"][f'branch_{resolution - 1 - i}'])
    


def main():
    # Excite the first dendrite
    excite_one_dendrite(0)

    # Excite the second dendrite
    excite_one_dendrite(2)

    calculate_morphology(resolution=10, secondary_stem_len=1, secondary_branch_len=100)

if __name__ == "__main__":
    main()
