package neuron

import (
	"fmt"
)

// A Net is a neural network consisting of a sequence of layers, each of which
// contains one or more Units.
type Net struct {
	Arch         []int
	InLayer      [](*InputUnit)
	HiddenLayers [][](*HiddenUnit)
	OutLayer     [](*OutputUnit)
}

// NewMLP constructs a new fully-connected network with the given architecture.
// TODO: Would be nice to also implement local connectivity, as in CNNs.
func NewMLP(arch []int) *Net {
	// Check for valid architecture
	numLayers := len(arch)
	if numLayers < 3 {
		panic(fmt.Sprintf("MLP architectures need >= 2 layers; got %d",
			numLayers))
	}
	for _, sz := range arch {
		if sz < 1 {
			panic(fmt.Sprintf("Each layer >= 1 unit; got %d", sz))
		}
	}

	n := Net{
		Arch:         make([]int, len(arch)),
		InLayer:      make([](*InputUnit), arch[0]),
		HiddenLayers: make([][](*HiddenUnit), numLayers-2),
		OutLayer:     make([](*OutputUnit), arch[numLayers-1]),
	}

	Logf(1, "Building a %d layer network.\n  Arch=%v\n", numLayers, arch)
	copy(n.Arch, arch)
	const idFormStr = "%03d_%06d"

	// Make input layer.
	ii := 0
	for jj := 0; jj < arch[ii]; jj++ {
		id := fmt.Sprintf(idFormStr, ii, jj)
		n.InLayer[jj] = NewInputUnit(id)
	}

	// Make Hidden layers.
	for ii := 1; ii < numLayers-1; ii++ {
		lH := make([](*HiddenUnit), arch[ii])
		for jj := 0; jj < arch[ii]; jj++ {
			id := fmt.Sprintf(idFormStr, ii, jj)
			lH[jj] = NewHiddenUnit(id)
		}
		n.HiddenLayers[ii-1] = lH
	}

	// Make output layer.
	ii = len(arch) - 1
	for jj := 0; jj < arch[ii]; jj++ {
		id := fmt.Sprintf(idFormStr, ii, jj)
		n.OutLayer[jj] = NewOutputUnit(id)
	}

	// Connect all the layers in a fully-connected pattern.
	for ii := 0; ii < numLayers-1; ii++ {
		for jj := 0; jj < arch[ii]; jj++ {
			for kk := 0; kk < arch[ii+1]; kk++ {
				switch ii {
				case 0:
					FeedIn(n.InLayer[jj], n.HiddenLayers[ii][kk])
				case numLayers - 2:
					FeedOut(n.HiddenLayers[ii-1][jj], n.OutLayer[kk])
				default:
					Connect(n.HiddenLayers[ii-1][jj], n.HiddenLayers[ii][kk])
				}
			}
		}
	}
	return &n
}

// Forward pass through the network.
func (n *Net) Forward(data []float64) (output []float64) {
	inDim := len(data)
	if inDim != n.Arch[0] {
		panic(fmt.Sprintf("Input dim (%d) not equal to number of input units (%d)",
			inDim, n.Arch[0]))
	}

	// Feed in.
	for ii, v := range data {
		n.InLayer[ii].Input <- v
	}

	outDim := n.Arch[len(n.Arch)-1]
	output = make([]float64, outDim)

	// Feed out.
	for ii := 0; ii < outDim; ii++ {
		output[ii] = <-n.OutLayer[ii].Output
	}
	return
}

// Backward pass through the network.
func (n *Net) Backward(grad []float64) (gradData []float64) {
	outDim := n.Arch[len(n.Arch)-1]
	gradDim := len(grad)
	if gradDim != outDim {
		panic(fmt.Sprintf("Grad dim (%d) not equal to number of output units (%d)",
			gradDim, outDim))
	}

	// Feed in (backward).
	for ii, v := range grad {
		n.OutLayer[ii].InputB <- v
	}

	inDim := n.Arch[0]
	gradData = make([]float64, inDim)

	// Feed out (backward).
	for ii := 0; ii < inDim; ii++ {
		gradData[ii] = <-n.InLayer[ii].OutputB
	}
	return
}

// Start running each unit's forward/backward/step loop concurrently.
func (n *Net) Start(train bool, updateFreq int, lr float64) {
	for _, u := range n.InLayer {
		go start(u, train, updateFreq, lr)
		Logf(2, "Start %s\n", u.ID)
	}
	for _, lH := range n.HiddenLayers {
		for _, u := range lH {
			go start(u, train, updateFreq, lr)
			Logf(2, "Start %s\n", u.ID)
		}
	}
	for _, u := range n.OutLayer {
		go start(u, train, updateFreq, lr)
		Logf(2, "Start %s\n", u.ID)
	}
}

// Start a unit's forward/backward/step loop. Note that forward and backward
// wait for inputs and grads from outside.
func start(u Unit, train bool, updateFreq int, lr float64) {
	step := 1
	for {
		u.Forward()
		if train {
			u.Backward()
			if updateFreq > 0 && step%updateFreq == 0 {
				u.Step(lr)
			}
		}
		step++
	}
}
