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

def main():
    # Excite the first dendrite
    excite_one_dendrite(0)

    # Excite the second dendrite
    excite_one_dendrite(2)

if __name__ == "__main__":
    main()