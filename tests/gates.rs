use num::Complex;
use qualcul::{ComplexMatrix, Gate};

#[test]
fn controlled_0_controls_1() {
	let mut expected_cx = ComplexMatrix::zero(4);
	expected_cx[(0, 0)] = Complex::from(1.0);
	expected_cx[(1, 1)] = Complex::from(1.0);
	expected_cx[(2, 3)] = Complex::from(1.0);
	expected_cx[(3, 2)] = Complex::from(1.0);
	let x = Gate::x();
	let cx = Gate::controlled(&x);
	let cx = Gate::map(&cx, &[0, 1]);
	assert!(cx.as_matrix().approx_eq(&expected_cx, 1e-6));
}

#[test]
fn controlled_1_controls_0() {
	let cx = Gate::controlled(&Gate::x());
	let reverse_cx = Gate::map(&cx, &[1, 0]);
	let cx = cx.as_matrix();
	let reverse_cx = reverse_cx.as_matrix();

	let mut expected_reverse_cx = ComplexMatrix::zero(4);
	expected_reverse_cx[(0, 0)] = Complex::from(1.0);
	expected_reverse_cx[(1, 3)] = Complex::from(1.0);
	expected_reverse_cx[(2, 2)] = Complex::from(1.0);
	expected_reverse_cx[(3, 1)] = Complex::from(1.0);
	assert!(reverse_cx.approx_eq(&expected_reverse_cx, 1e-6));

	let swap = cx * reverse_cx;
	let swap = &swap * cx;
	let expected_swap = Gate::swap();
	let expected_swap = expected_swap.as_matrix();
	assert!(swap.approx_eq(&expected_swap, 1e-6));
}

#[test]
// Used qiskit to verify, however the endian-ness is different so it's equivalent to cx(2, 0)
fn controlled_0_controls_2() {
	let mut expected_cx = ComplexMatrix::zero(8);
	expected_cx[(0, 0)] = Complex::from(1.0);
	expected_cx[(1, 1)] = Complex::from(1.0);
	expected_cx[(2, 2)] = Complex::from(1.0);
	expected_cx[(3, 3)] = Complex::from(1.0);
	expected_cx[(5, 4)] = Complex::from(1.0);
	expected_cx[(4, 5)] = Complex::from(1.0);
	expected_cx[(7, 6)] = Complex::from(1.0);
	expected_cx[(6, 7)] = Complex::from(1.0);
	let x = Gate::x();
	let cx = Gate::controlled(&x);
	let cx = Gate::map(&cx, &[0, 2, 1]);

	dbg!(&cx.as_matrix());
	dbg!(&expected_cx);

	assert!(cx.as_matrix().approx_eq(&expected_cx, 1e-6));
}

#[test]
// Source: https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu/4254#4254
fn controlled_2_controls_0() {
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
	let cx = Gate::controlled(&x);
	let cx = Gate::map(&cx, &[2, 0, 1]);

	dbg!(&cx.as_matrix());
	dbg!(&expected_cx);

	assert!(cx.as_matrix().approx_eq(&expected_cx, 1e-6));
}

#[test]
fn ccx() {
	let mut expected_ccx = ComplexMatrix::identity(8);
	expected_ccx[(6, 7)] = Complex::from(1.0);
	expected_ccx[(7, 6)] = Complex::from(1.0);
	expected_ccx[(6, 6)] = Complex::from(0.0);
	expected_ccx[(7, 7)] = Complex::from(0.0);
	let x = Gate::x();
	let cx = Gate::controlled(&x);
	let ccx = Gate::controlled(&cx);
	let ccx = Gate::map(&ccx, &[0, 1, 2]);
	assert!(ccx.as_matrix().approx_eq(&expected_ccx, 1e-6));
}

#[test]
fn fredkin() {
	let mut expected_fredkin = ComplexMatrix::identity(8);
	expected_fredkin[(5, 6)] = Complex::from(1.0);
	expected_fredkin[(6, 5)] = Complex::from(1.0);
	expected_fredkin[(5, 5)] = Complex::from(0.0);
	expected_fredkin[(6, 6)] = Complex::from(0.0);
	let swap = Gate::swap();
	let fredkin = Gate::controlled(&swap);
	let fredkin = Gate::map(&fredkin, &[0, 1, 2]);
	assert!(fredkin.as_matrix().approx_eq(&expected_fredkin, 1e-6));
}
