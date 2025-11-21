use num::Complex;
use qualcul::{ComplexMatrix, Gate};

#[test]
fn controlled_adjacent() {
	let mut expected_cx = ComplexMatrix::zero(4);
	expected_cx[(0, 0)] = Complex::from(1.0);
	expected_cx[(1, 1)] = Complex::from(1.0);
	expected_cx[(2, 3)] = Complex::from(1.0);
	expected_cx[(3, 2)] = Complex::from(1.0);
	let x = Gate::x();
	let cx = Gate::controlled_adjacent(&x);
	assert!(cx.as_matrix().approx_eq(&expected_cx, 1e-6));
}

#[test]
fn controlled_adjacent_reversed() {
	let cx = Gate::controlled(&Gate::x(), 0, 1, 2);
	let cx = cx.as_matrix();
	let reverse_cx = Gate::controlled_adjacent_reversed(&Gate::x());
	let reverse_cx = reverse_cx.as_matrix();
	let swap = cx * reverse_cx;
	let swap = &swap * cx;
	assert!(swap.approx_eq(Gate::swap().as_matrix(), 1e-6));
}

#[test]
// Source: https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu/4254#4254
fn controlled_non_adjacent() {
	let mut expected_cx = ComplexMatrix::zero(8);
	expected_cx[(0, 0)] = Complex::from(1.0);
	expected_cx[(1, 5)] = Complex::from(1.0);
	expected_cx[(2, 2)] = Complex::from(1.0);
	expected_cx[(3, 7)] = Complex::from(1.0);
	expected_cx[(4, 4)] = Complex::from(1.0);
	expected_cx[(5, 1)] = Complex::from(1.0);
	expected_cx[(6, 6)] = Complex::from(1.0);
	expected_cx[(7, 3)] = Complex::from(1.0);
	let x = Gate::x();
	let cx = Gate::controlled(&x, 2, 0, 3);
	assert!(cx.as_matrix().approx_eq(&expected_cx, 1e-6));
}
