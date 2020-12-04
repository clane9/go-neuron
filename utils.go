package neuron

import (
	"fmt"
	"time"
)

// Verbosity is the global verbosity level.
var Verbosity = 3

// logf logs output if it exceeds the global verbosity level.
func logf(level int, format string, a ...interface{}) (n int, err error) {
	t := time.Now()
	prefix := fmt.Sprintf("(%d) (%s) ", level, t.Format("15:04:05.999"))
	format = prefix + format
	if level <= Verbosity {
		n, err = fmt.Printf(format, a...)
	}
	return
}
