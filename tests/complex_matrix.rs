use num::Complex;
use qualcul::ComplexMatrix;

#[test]
fn kronecker_product() {
	let a = ComplexMatrix::from(&vec![
		vec![Complex::from(1.0), Complex::from(2.0)],
		vec![Complex::from(3.0), Complex::from(1.0)],
	]);
	let b = ComplexMatrix::from(&vec![
		vec![Complex::from(0.0), Complex::from(3.0)],
		vec![Complex::from(2.0), Complex::from(1.0)],
	]);
	let a_prod_b = a.kronecker_product(&b);

	let expected = ComplexMatrix::from(&vec![
		vec![Complex::from(0.0), Complex::from(3.0), Complex::from(0.0), Complex::from(6.0)],
		vec![Complex::from(2.0), Complex::from(1.0), Complex::from(4.0), Complex::from(2.0)],
		vec![Complex::from(0.0), Complex::from(9.0), Complex::from(0.0), Complex::from(3.0)],
		vec![Complex::from(6.0), Complex::from(3.0), Complex::from(2.0), Complex::from(1.0)],
	]);

	assert!(expected.approx_eq(&a_prod_b, 1e-10));
}
