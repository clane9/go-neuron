package neuron

import (
	"fmt"
)

// A Net is a neural network consisting of a sequence of layers, each of which
// contains one or more Units. Arch defines the layer sizes. Layers points to
// each of the units in each of the layers.
type Net struct {
	// Size of each layer
	Arch []int
	// Pointers to the units in each layer
	Layers   [][](*Unit)
	stepDone chan int
}

// NewMLP constructs a new fully-connected network with the given architecture.
func NewMLP(arch []int, opt Optimizer) *Net {
	// Check for valid architecture
	numLayers := len(arch)
	if numLayers < 3 {
		// TODO: These should probably be errors, not panics. Also, add error
		// handling elsewhere as needed.
		panic(fmt.Sprintf("MLP architectures need >= 2 layers; got %d",
			numLayers))
	}
	for _, sz := range arch {
		if sz < 1 {
			panic(fmt.Sprintf("Each layer >= 1 unit; got %d", sz))
		}
	}

	n := Net{
		Arch:     make([]int, len(arch)),
		Layers:   make([][](*Unit), numLayers),
		stepDone: make(chan int),
	}

	logf(1, "Building a %d layer network.\n  Arch=%v\n", numLayers, arch)
	copy(n.Arch, arch)

	// Make layers.
	const idFormStr = "%03d_%06d"
	var id string
	var u *Unit
	for ii := 0; ii < numLayers; ii++ {
		l := make([]*Unit, arch[ii])
		for jj := 0; jj < arch[ii]; jj++ {
			id = fmt.Sprintf(idFormStr, ii, jj)
			switch ii {
			case 0:
				// Need a new opt for each unit so that each gets their own buffer data.
				u = newInputUnit(id, opt.New(), n.stepDone)
			case numLayers - 1:
				u = newOutputUnit(id, opt.New(), n.stepDone)
			default:
				u = newHiddenUnit(id, opt.New(), n.stepDone)
			}
			l[jj] = u
		}
		n.Layers[ii] = l
	}

	// Connect all the layers in a fully-connected pattern.
	for ii := 0; ii < numLayers-1; ii++ {
		for _, u1 := range n.Layers[ii] {
			for _, u2 := range n.Layers[ii+1] {
				u1.connect(u2)
			}
		}
	}
	return &n
}

// Forward pass through the network. The input is a single data sample.
func (n *Net) Forward(data []float64) (output []float64) {
	inDim := len(data)
	if inDim != n.Arch[0] {
		panic(fmt.Sprintf("Input dim (%d) not equal to number of input units (%d)",
			inDim, n.Arch[0]))
	}

	logf(2, "MLP Forward\n")

	// Feed in.
	for ii, v := range data {
		n.Layers[0][ii].input <- signal{id: inputID, value: v}
	}

	numLayers := len(n.Arch)
	outDim := n.Arch[numLayers-1]
	output = make([]float64, outDim)

	// Feed out.
	var s signal
	for ii := 0; ii < outDim; ii++ {
		s = <-n.Layers[numLayers-1][ii].output[outputID]
		output[ii] = s.value
	}
	return
}

// Backward pass a loss gradient through the network. Input grad should be a
// gradient with respect to each of the network outputs.
func (n *Net) Backward(grad []float64) {
	outDim := n.Arch[len(n.Arch)-1]
	gradDim := len(grad)
	if gradDim != outDim {
		panic(fmt.Sprintf("Grad dim (%d) not equal to number of output units (%d)",
			gradDim, outDim))
	}

	logf(2, "MLP Backward\n")

	// Feed in (backward).
	numLayers := len(n.Arch)
	for ii, v := range grad {
		n.Layers[numLayers-1][ii].inputB <- signal{id: inputID, value: v}
	}

	// Wait for all units to finish backward and step to avoid a race.
	n.sync()
}

// sync waits for all units to complete their forward/backward/step sequence.
func (n *Net) sync() {
	totalUnits := 0
	for _, v := range n.Arch {
		totalUnits += v
	}
	for ii := 0; ii < totalUnits; ii++ {
		<-n.stepDone
	}
}

// Start running each unit's forward/backward/step loop concurrently. Neuron
// weights and biases are updated every updateFreq iterations. By setting
// updateFreq > 1, we can simulate mini-batch optimization.
func (n *Net) Start(train bool, updateFreq int) {
	for _, l := range n.Layers {
		for _, u := range l {
			go u.start(train, updateFreq)
			logf(2, "Start %s\n", u.ID)
		}
	}
}
