use num::Complex;
use qualcul::{ComplexMatrix, Gate};

#[test]
fn controlled() {
	let x = Gate::x();

	let cx = Gate::controlled(&x);
	let mut expected_cx = ComplexMatrix::zero(4);
	expected_cx[(0, 0)] = Complex::from(1.0);
	expected_cx[(1, 1)] = Complex::from(1.0);
	expected_cx[(2, 3)] = Complex::from(1.0);
	expected_cx[(3, 2)] = Complex::from(1.0);
	assert!(cx.as_matrix().approx_eq(&expected_cx, 1e-6));

	let ccx = Gate::controlled(&cx);
	let mut expected_ccx = ComplexMatrix::identity(8);
	expected_ccx[(6, 6)] = Complex::from(0.0);
	expected_ccx[(7, 7)] = Complex::from(0.0);
	expected_ccx[(6, 7)] = Complex::from(1.0);
	expected_ccx[(7, 6)] = Complex::from(1.0);
	assert!(ccx.as_matrix().approx_eq(&expected_ccx, 1e-6));
}

#[test]
fn reverse_controlled() {
	let cx = Gate::controlled(&Gate::x());
	let cx = cx.as_matrix();
	let reverse_cx = Gate::reverse_controlled(&Gate::x());
	let reverse_cx = reverse_cx.as_matrix();
	let swap = cx * reverse_cx;
	let swap = &swap * cx;
	assert!(swap.approx_eq(Gate::swap().as_matrix(), 1e-6));
}
