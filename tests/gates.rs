use num::Complex;
use qualcul::{ComplexMatrix, Gate};

#[test]
fn controlled_0_controls_1() {
	let mut expected_cx = ComplexMatrix::zero(4);
	expected_cx[(0, 0)] = Complex::from(1.0);
	expected_cx[(1, 1)] = Complex::from(1.0);
	expected_cx[(2, 3)] = Complex::from(1.0);
	expected_cx[(3, 2)] = Complex::from(1.0);
	let cx = Gate::x().on(1).control(vec![0]).into_matrix();

	dbg!(&cx);
	dbg!(&expected_cx);
	assert!(cx.approx_eq(&expected_cx, 1e-6));
}

#[test]
fn controlled_1_controls_0() {
	let cx = Gate::x().on(1).control(vec![0]).into_matrix();
	let reverse_cx = Gate::x().on(0).control(vec![1]).into_matrix();

	let mut expected_reverse_cx = ComplexMatrix::zero(4);
	expected_reverse_cx[(0, 0)] = Complex::from(1.0);
	expected_reverse_cx[(1, 3)] = Complex::from(1.0);
	expected_reverse_cx[(2, 2)] = Complex::from(1.0);
	expected_reverse_cx[(3, 1)] = Complex::from(1.0);
	dbg!(&reverse_cx);
	dbg!(&expected_reverse_cx);
	assert!(reverse_cx.approx_eq(&expected_reverse_cx, 1e-6));

	let swap = &cx * &reverse_cx;
	let swap = &swap * &cx;

	let expected_swap = Gate::swap();
	let expected_swap = expected_swap.as_matrix();

	dbg!(&swap);
	dbg!(&expected_swap);
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

	let cx = Gate::x().on(2).control(vec![0]).into_matrix();

	dbg!(&cx);
	dbg!(&expected_cx);
	assert!(cx.approx_eq(&expected_cx, 1e-6));
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

	let cx = Gate::x().on(0).control(vec![2]).into_matrix();

	dbg!(&cx);
	dbg!(&expected_cx);
	assert!(cx.approx_eq(&expected_cx, 1e-6));
}

#[test]
fn ccx() {
	let mut expected_ccx = ComplexMatrix::identity(8);
	expected_ccx[(6, 7)] = Complex::from(1.0);
	expected_ccx[(7, 6)] = Complex::from(1.0);
	expected_ccx[(6, 6)] = Complex::from(0.0);
	expected_ccx[(7, 7)] = Complex::from(0.0);
	let ccx = Gate::x().on(2).control(vec![0, 1]).into_matrix();
	assert!(ccx.approx_eq(&expected_ccx, 1e-6));
}

#[test]
fn fredkin() {
	let mut expected_fredkin = ComplexMatrix::identity(8);
	expected_fredkin[(5, 6)] = Complex::from(1.0);
	expected_fredkin[(6, 5)] = Complex::from(1.0);
	expected_fredkin[(5, 5)] = Complex::from(0.0);
	expected_fredkin[(6, 6)] = Complex::from(0.0);
	let fredkin = Gate::swap().on_qubits(vec![1, 2]).control(vec![0]).into_matrix();
	assert!(fredkin.approx_eq(&expected_fredkin, 1e-6));
}

#[test]
fn t_is_sqrt_of_s() {
	let s = Gate::s();
	let t = Gate::t();
	let t_squared = t.as_matrix() * t.as_matrix();
	assert!(t_squared.approx_eq(s.as_matrix(), 1e-6));
}

#[test]
fn s_is_sqrt_of_z() {
	let z = Gate::z();
	let s = Gate::s();
	let s_squared = s.as_matrix() * s.as_matrix();
	assert!(s_squared.approx_eq(z.as_matrix(), 1e-6));
}
