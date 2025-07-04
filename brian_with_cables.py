from brian2 import *
prefs.codegen.target = "numpy"   # Optional; makes things run without Cython

def excite_one_dendrite(dendrite):
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


def excite_both_dendrites(time1, time2):
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
        threshold='v > 0.99*mV',
        refractory='v > -10*mV',
        Cm=1*uF/cm**2,
        Ri=100*ohm*cm,
        method='exponential_euler',
    )

    neuron.v = -65*mV
    neuron.gl = 0.0003*siemens/cm**2
    neuron.El = -65*mV

    # Input group: spike train at given times
    spiketimes1 = [time1]*ms
    input1 = SpikeGeneratorGroup(1, [0]*len(spiketimes1), spiketimes1)

    spiketimes2 = [time2]*ms
    input2 = SpikeGeneratorGroup(1, [0]*len(spiketimes2), spiketimes2)

    # Connect input to the chosen dendrite
    syn1 = Synapses(input1, neuron, on_pre='v_post += 0.5*mV')
    syn1.connect(i=0, j=0)

    syn2 = Synapses(input2, neuron, on_pre='v_post += 0.5*mV')
    syn2.connect(i=0, j=2)

    M = StateMonitor(neuron, 'v', record=[0, 1, 2])
    run(500*ms)

    plot(M.t/ms, M.v[0]/mV, label='dendL')
    plot(M.t/ms, M.v[1]/mV, label='soma')
    plot(M.t/ms, M.v[2]/mV, label='dendR')
    xlabel('Time (ms)')
    ylabel('v (mV)')
    legend()
    title(f'Stimulus on dendrites')
    show()

def main():
    # Excite the first dendrite
    excite_both_dendrites(time1=100, time2=200)

    # Excite the second dendrite
    #excite_one_dendrite(2)

if __name__ == "__main__":
    main()