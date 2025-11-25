use num::Complex;
use qualcul::{
	ComplexMatrix, Gate,
	circuit::{Circuit, StateVector},
	state::Ket,
};

#[test]
fn epr_pair_matrix() {
	let circuit = Circuit::new()
		.then(Gate::map(&Gate::h(), &[0, 1]))
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 1]));
	let matrix = circuit.as_matrix();

	let mut expected_matrix = ComplexMatrix::zero(4);
	expected_matrix[(0, 0)] = Complex::from(1.0);
	expected_matrix[(0, 2)] = Complex::from(1.0);
	expected_matrix[(1, 1)] = Complex::from(1.0);
	expected_matrix[(1, 3)] = Complex::from(1.0);
	expected_matrix[(2, 1)] = Complex::from(1.0);
	expected_matrix[(2, 3)] = Complex::from(-1.0);
	expected_matrix[(3, 0)] = Complex::from(1.0);
	expected_matrix[(3, 2)] = Complex::from(-1.0);
	expected_matrix *= Complex::from(1.0 / 2.0f64.sqrt());

	assert!(matrix.approx_eq(&expected_matrix, 1e-6));
}

#[test]
fn epr_pair_run() {
	let circuit = Circuit::new()
		.then(Gate::map(&Gate::h(), &[0, 1]))
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 1]));
	let initial_state = StateVector::from_ket(&Ket::base(0b00, 4));
	let final_state = circuit.run(initial_state);
	let expected_state = StateVector::from_ket(&Ket::bell_phi_plus());
	assert!(final_state.approx_eq(&expected_state, 1e-6));
}

#[test]
fn ghz_n_run() {
	for n in 0..=5 {
		dbg!(n);

		// h on qubit 0
		let mappings: Vec<_> = (0..=n).collect();
		let mut circuit = Circuit::new().then(Gate::map(&Gate::h(), &mappings));

		// cx with qubit i controlling qubit i+1
		for i in 0..n {
			let mut mappings = mappings.clone();
			mappings.swap(0, i);
			mappings.swap(1, i + 1);
			circuit = circuit.then(Gate::map(&Gate::controlled(&Gate::x()), &mappings));
		}

		let nb_dimensions = 2 << n; // 2^(n+1)
		let ket_0s = Ket::base(0, nb_dimensions);
		let ket_1s = Ket::base((2 << n) - 1, nb_dimensions);

		let initial_state = StateVector::from_ket(&ket_0s);
		dbg!(&initial_state);

		let final_state = circuit.run(initial_state);
		let expected_state = StateVector::from_ket(&Ket::add_and_normalize(&ket_0s, &ket_1s));
		dbg!(&final_state);
		dbg!(&expected_state);
		assert!(final_state.approx_eq(&expected_state, 1e-6));

		let possible_outcomes = final_state.possible_outcomes();
		assert!(possible_outcomes.len() == 2);
		let (outcome1, probability1) = &possible_outcomes[0];
		let (outcome2, probability2) = &possible_outcomes[1];
		assert!(*outcome1 == ket_0s || *outcome1 == ket_1s);
		assert!(*outcome2 == ket_0s || *outcome2 == ket_1s);
		assert!((*probability1 - 0.5f64).abs() <= 1e-6);
		assert!((*probability2 - 0.5f64).abs() <= 1e-6);
	}
}
