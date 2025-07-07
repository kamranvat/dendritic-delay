# simple brian2 implementation of the sound localization model proposed by Jeffress (1948).

from brian2 import *
prefs.codegen.target = "numpy"   # Use numpy instead of Cython to avoid compilation warnings

def calculate_morphology(resolution=10, max_dendrite_len=100):
    # calculate a morphology with a given resolution and delay line length
    # setup: 10 somas with 2 dendrites each
    # left dendrites increase linearly, right dendrites decrease linearly
    # all left synapses get shared input, all right synapses get shared input
    
    start_scope()
    
    # get the lengths of the dendrites (linearly spaced)
    lengths = np.linspace(0, max_dendrite_len, resolution+2)[1:-1]  # avoid zero length dendrites
    
    morphologies = []
    
    # build each soma with its dendrites
    for i in range(resolution):
        soma = Soma(diameter=30*um)
        
        # left dendrite increases, right dendrite decreases linearly
        left_length = lengths[i]   
        right_length = lengths[resolution - 1 - i]
        
        # add dendrites to soma
        soma.left = Cylinder(length=left_length*um, diameter=2*um, n=1)
        soma.right = Cylinder(length=right_length*um, diameter=2*um, n=1)
        
        morphologies.append(soma)
    
    eqs = '''
    Im = gl * (El - v) : amp/meter**2
    gl : siemens/meter**2
    El : volt
    '''
    
    # create one spatial neuron per morphology
    neurons = []
    for morph in morphologies:
        neuron = SpatialNeuron(
            morphology=morph,
            model=eqs,
            Cm=1*uF/cm**2,
            Ri=100*ohm*cm,
            method='exponential_euler',
        )
        
        # neuron params
        neuron.v = -65*mV
        neuron.gl = 0.0003*siemens/cm**2
        neuron.El = -65*mV
        
        neurons.append(neuron)
    
    # create shared inputs
    input_left = SpikeGeneratorGroup(1, [0], [50*ms])
    input_right = SpikeGeneratorGroup(1, [0], [75*ms])
    
    # store synapses to keep them in scope
    synapses_left = []
    synapses_right = []
    
    # create synapses for each neuron individually
    for i, neuron in enumerate(neurons):
        # left input connects to left dendrite (compartment 0)
        syn_left = Synapses(input_left, neuron, on_pre='v_post += 2*mV')
        syn_left.connect(i=0, j=0)  # connect to left dendrite
        synapses_left.append(syn_left)
        
        # right input connects to right dendrite (compartment 2)
        syn_right = Synapses(input_right, neuron, on_pre='v_post += 2*mV') 
        syn_right.connect(i=0, j=2)  # connect to right dendrite
        synapses_right.append(syn_right)
    
    # setup monitoring for all neurons
    monitors = []
    for neuron in neurons:
        monitor = StateMonitor(neuron, 'v', record=True)
        monitors.append(monitor)
    
    # create network with all objects
    network_objects = [input_left, input_right] + neurons + synapses_left + synapses_right + monitors
    net = Network(*network_objects)
    
    # run simulation
    net.run(200*ms)
    
    # plot results - show soma voltages for first few neurons
    plt.figure(figsize=(12, 8))
    
    for i in range(resolution): 
        # soma is compartment 1 in each neuron
        plt.plot(monitors[i].t/ms, monitors[i].v[1]/mV, label=f'Soma {i} (L={lengths[i]:.1f}, R={lengths[resolution-1-i]:.1f})')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.title('Voltage traces for somas in new morphology')
    plt.tight_layout()
    plt.show()
    
    print(f"Simulation completed. Plotted {min(5, resolution)} soma voltage traces.")
    print("Dendrite lengths (left, right) for each neuron:")
    for i in range(resolution):
        print(f"  Neuron {i}: L={lengths[i]:.1f}μm, R={lengths[resolution-1-i]:.1f}μm")
    
    return neurons, monitors


def main():
    # Test 
    neurons, monitors = calculate_morphology(resolution=10, max_dendrite_len=100)

if __name__ == "__main__":
    main()
