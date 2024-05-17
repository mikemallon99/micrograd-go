package microgradgo

import (
	"fmt"
	"testing"
)

func TestValue(t *testing.T) {
	val1 := MakeValue(1.0, []*Value{}, "")
	if val1.Data != 1.0 {
		t.Fatalf("Value isnt 1.0 for some reason")
	}

	val2 := MakeValue(2.0, []*Value{}, "")
	add_vals := val1.add(val2)
	if add_vals.Data != 3.0 {
		t.Fatalf("Value isnt 3.0 for some reason")
	}

	add_vals.backward()
	if add_vals.Grad != 1 {
		t.Fatalf("add_vals.Grad isnt 1.0")
	}
	if val1.Grad != 1 {
		t.Fatalf("add_vals.Grad isnt 1.0")
	}
	if val2.Grad != 1 {
		t.Fatalf("add_vals.Grad isnt 1.0")
	}
}

func TestNeuron(t *testing.T) {
	neuron := MakeNeuron(5, false)
	x := []*Value{
		MakeValue(1.0, []*Value{}, ""),
		MakeValue(2.0, []*Value{}, ""),
		MakeValue(3.0, []*Value{}, ""),
		MakeValue(4.0, []*Value{}, ""),
		MakeValue(5.0, []*Value{}, ""),
	}

	fmt.Println(neuron.String())
	params := neuron.parameters()
	for i := range params {
		fmt.Println(params[i].String())
	}

	act := neuron.forward(x)
	fmt.Println(act.String())
}
