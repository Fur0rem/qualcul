use std::f64::consts::PI;

use num::Complex;
use qualcul::algorithms::{qft_circuit, qft_matrix};
use qualcul::{
	ComplexMatrix, Gate, Operation,
	circuit::{Circuit, StateVector},
	state::Ket,
};

const NB_RANDOM_TESTS: usize = 20;

#[test]
fn quantum_teleportation() {
	let circuit = Circuit::new(3)
		.then(Gate::map(&Gate::h(), &[2, 0, 1]))
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[2, 1, 0]))
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 1, 2]))
		.then(Gate::map(&Gate::h(), &[0, 1, 2]))
		.then_op(Operation::Measure(0))
		.then_op(Operation::Measure(1))
		.then_op(Operation::ClassicalControl {
			look_up_table: {
				let mut table = Vec::new();
				table.push((vec![(1, 0)], Gate::map(&Gate::none(), &[0, 1, 2])));
				table.push((vec![(1, 1)], Gate::map(&Gate::x(), &[2, 0, 1])));
				table
			},
		})
		.then_op(Operation::ClassicalControl {
			look_up_table: {
				let mut table = Vec::new();
				table.push((vec![(0, 0)], Gate::map(&Gate::none(), &[0, 1, 2])));
				table.push((vec![(0, 1)], Gate::map(&Gate::z(), &[2, 0, 1])));
				table
			},
		});

	for _ in 0..NB_RANDOM_TESTS {
		let alice_qubit = Ket::random(2);
		let shared_qubit = Ket::base(0b0, 2);
		let bob_qubit = Ket::base(0b0, 2);

		let initial_state = StateVector::from_qubits(&[&alice_qubit, &shared_qubit, &bob_qubit]);
		let final_state = circuit.run(&initial_state);

		let bob_new_qubit = final_state.extract_state_of_single_qubit(2);

		dbg!(&alice_qubit);
		dbg!(&bob_new_qubit);
		assert!(alice_qubit.approx_eq(&bob_new_qubit, 1e-6));
	}
}

#[test]
fn z_error_correction() {
	let x = Gate::x();
	let x = x.as_matrix();
	let id = ComplexMatrix::identity(2);

	// 3 physical qubits, 2 error correction qubits
	let circuit = Circuit::new(5)
		// Encode
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 1, 2, 3, 4])) // 0 -> 1
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 2, 1, 3, 4])) // 0 -> 2
		// Apply h gates
		.then(Gate::map(&Gate::h(), &[0, 1, 2, 3, 4])) // h on qubit 0
		.then(Gate::map(&Gate::h(), &[1, 0, 2, 3, 4])) // h on qubit 1
		.then(Gate::map(&Gate::h(), &[2, 0, 1, 3, 4])) // h on qubit 2
		// Error (Z gates only)
		.then(Gate::map(&Gate::z(), &[1, 0, 2, 3, 4])) // Z error on qubit 1
		// Apply h gates again
		.then(Gate::map(&Gate::h(), &[0, 1, 2, 3, 4])) // h on qubit 0
		.then(Gate::map(&Gate::h(), &[1, 0, 2, 3, 4])) // h on qubit 1
		.then(Gate::map(&Gate::h(), &[2, 0, 1, 3, 4])) // h on qubit 2
		// Repair
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 3, 1, 2, 4])) // 0 -> 3
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[1, 3, 0, 2, 4])) // 1 -> 3
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[1, 4, 0, 2, 3])) // 1 -> 4
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[2, 4, 0, 1, 3])) // 2 -> 4
		.then_op(Operation::Measure(3)) // Measure qubit 3
		.then_op(Operation::Measure(4)) // Measure qubit 4
		.then_op(Operation::ClassicalControl {
			look_up_table: vec![
				(
					vec![(3, 0), (4, 0)],
					Gate::map(&Gate::from(id.kronecker_product(&id).kronecker_product(&id)), &[0, 1, 2, 3, 4]),
				),
				(
					vec![(3, 0), (4, 1)],
					Gate::map(&Gate::from(id.kronecker_product(&id).kronecker_product(&x)), &[0, 1, 2, 3, 4]),
				),
				(
					vec![(3, 1), (4, 0)],
					Gate::map(&Gate::from(x.kronecker_product(&id).kronecker_product(&id)), &[0, 1, 2, 3, 4]),
				),
				(
					vec![(3, 1), (4, 1)],
					Gate::map(&Gate::from(id.kronecker_product(&x).kronecker_product(&id)), &[0, 1, 2, 3, 4]),
				),
			],
		}) // Kraus operator
		// Decode
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 1, 2, 3, 4])) // 0 -> 1
		.then(Gate::map(&Gate::controlled(&Gate::x()), &[0, 2, 1, 3, 4])); // 0 -> 2

	for _ in 0..NB_RANDOM_TESTS {
		let q0 = Ket::random(2);
		let q1 = Ket::base(0b0, 2);
		let q2 = Ket::base(0b0, 2);
		let q3 = Ket::base(0b0, 2);
		let q4 = Ket::base(0b0, 2);
		let initial_state = StateVector::from_qubits(&[&q0, &q1, &q2, &q3, &q4]);
		let final_state = circuit.run(&initial_state);
		let recovered_q0 = final_state.extract_state_of_single_qubit(0);
		dbg!(&q0);
		dbg!(&recovered_q0);
		assert!(q0.approx_eq(&recovered_q0, 1e-4));
	}
}

#[test]
fn qft_1_qubit() {
	// QFT with 1 qubit is just H
	let given_circuit = Circuit::new(1).then(Gate::map(&Gate::h(), &[0]));
	let built_circuit = qft_circuit(1);
	let actual_matrix = qft_matrix(1);

	dbg!(&actual_matrix);
	dbg!(&given_circuit.as_matrix().unwrap());
	dbg!(&built_circuit.as_matrix().unwrap());

	assert!(given_circuit.as_matrix().unwrap().approx_eq(&actual_matrix, 1e-6));
	assert!(built_circuit.as_matrix().unwrap().approx_eq(&actual_matrix, 1e-6));
}

#[test]
fn qft_2_qubit() {
	// 2-qubit QFT circuit:
	// - H on qubit 0
	// - controlled R2 (control qubit 1 -> target qubit 0)
	// - H on qubit 1
	// - swap qubit 0 and 1
	let mut circuit = Circuit::new(2);
	circuit = circuit
		// H on 0
		.then(Gate::map(&Gate::h(), &[0, 1]))
		// Controlled R2 (1 -> 0)
		.then({
			let angle = 2.0 * PI / 4.0; // R2: 2pi / 2^2
			let r2 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			let phase_gate = Gate::from(r2);
			Gate::map(&Gate::controlled(&phase_gate), &[1, 0]) // 1 controls 0
		})
		// H on 1
		.then(Gate::map(&Gate::h(), &[1, 0]))
		// Swap 0 <-> 1
		.then(Gate::map(&Gate::swap(), &[0, 1]));

	let reference_matrix = qft_matrix(2);

	dbg!(&reference_matrix);
	dbg!(&circuit.as_matrix().unwrap());
	assert!(reference_matrix.approx_eq(&circuit.as_matrix().unwrap(), 1e-6));
}

#[test]
fn qft_3_qubit() {
	// 3-qubit QFT circuit:
	// - H on qubit 0
	// - controlled R2 (control qubit 1 -> target qubit 0)
	// - controlled R3 (control qubit 2 -> target qubit 0)
	// - H on qubit 1
	// - controlled R2 (control qubit 2 -> target qubit 1)
	// - H on qubit 2
	// - swap qubit 0 and 2
	let mut circuit = Circuit::new(3);
	circuit = circuit
		// H on qubit 0
		.then(Gate::map(&Gate::h(), &[0, 1, 2]))
		// Controlled R2: (1 -> 0)
		.then({
			let angle = 2.0 * PI / 4.0; // R2: 2pi / 2^2
			let r2 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			let phase_gate = Gate::from(r2);
			Gate::map(&Gate::controlled(&phase_gate), &[1, 0, 2]) // 1 controls 0
		})
		// Controlled R3: (2 -> 0)
		.then({
			let angle = 2.0 * PI / 8.0; // R3: 2pi / 2^3
			let r3 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			let phase_gate = Gate::from(r3);
			Gate::map(&Gate::controlled(&phase_gate), &[2, 0, 1]) // 2 controls 0
		})
		// H on qubit 1
		.then(Gate::map(&Gate::h(), &[1, 0, 2]))
		// Controlled R2: (2 -> 1)
		.then({
			let angle = 2.0 * PI / 4.0; // R2: 2pi / 2^2
			let r2 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			let phase_gate = Gate::from(r2);
			Gate::map(&Gate::controlled(&phase_gate), &[2, 1, 0]) // 2 controls 1
		})
		// H on qubit 2
		.then(Gate::map(&Gate::h(), &[2, 1, 0]))
		// Swap 0 <-> 2
		.then(Gate::map(&Gate::swap(), &[0, 2, 1]));

	let reference_matrix = qft_matrix(3);

	dbg!(&reference_matrix);
	dbg!(&circuit.as_matrix().unwrap());
	assert!(reference_matrix.approx_eq(&circuit.as_matrix().unwrap(), 1e-6));
}

#[test]
fn qft() {
	let frequency = 5;

	for nb_qubits in 4..=8 {
		let dimension = 2_usize.pow(nb_qubits as u32);

		// Check if circuit and matrix match
		let circuit = qft_circuit(nb_qubits);
		let matrix = qft_matrix(nb_qubits as u32);
		dbg!(&matrix);
		dbg!(&circuit.as_matrix().unwrap());
		assert!(matrix.approx_eq(&circuit.as_matrix().unwrap(), 1e-6));

		// Prepare state for a wave with given frequency
		let norm = 1.0 / (dimension as f64).sqrt();
		let mut amplitudes = Vec::with_capacity(dimension);
		for x in 0..dimension {
			let angle = -2.0 * PI * (x as f64) / frequency as f64;
			amplitudes.push(Complex::from_polar(norm, angle));
		}
		let state = StateVector::from_vec(amplitudes);

		// Check if total probability is 1
		let total_prob: f64 = (0..dimension).map(|i| state[i].norm_sqr()).sum();
		assert!((total_prob - 1.0).abs() < 1e-12);
		dbg!(&state);

		// Check if the final state gives the closest frequency
		let final_state = circuit.run(&state);
		let found_frequency = 2_usize.pow(nb_qubits as u32) as f64 / final_state.most_likely_outcome() as f64;
		dbg!(&final_state);
		dbg!(&found_frequency);
		assert!(found_frequency.round_ties_even() == frequency as f64);
	}
}
