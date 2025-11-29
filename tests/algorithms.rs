use std::f64::consts::PI;

use num::Complex;
use qualcul::Gate;
use qualcul::algorithms::{qft_circuit, qft_matrix};
use qualcul::backend::dense_cpu::DenseCPUBackend;
use qualcul::backend::{Backend, Program};
use qualcul::{
	ComplexMatrix, QuantumOperation,
	circuit::{Circuit, StateVector},
	state::Ket,
};

const NB_RANDOM_TESTS: usize = 20;

#[test]
fn quantum_teleportation() {
	let circuit = Circuit::new(3)
		.then(Gate::h().on(2))
		.then(Gate::x().on(1).control(vec![2]))
		.then(Gate::x().on(1).control(vec![0]))
		.then(Gate::h().on(0))
		.then_op(QuantumOperation::Measure(0))
		.then_op(QuantumOperation::Measure(1))
		.then_op(QuantumOperation::ClassicalControl {
			look_up_table: {
				let mut table = Vec::new();
				table.push((vec![(1, 0)], Gate::none().on(0)));
				table.push((vec![(1, 1)], Gate::x().on(2)));
				table
			},
		})
		.then_op(QuantumOperation::ClassicalControl {
			look_up_table: {
				let mut table = Vec::new();
				table.push((vec![(0, 0)], Gate::none().on(0)));
				table.push((vec![(0, 1)], Gate::z().on(2)));
				table
			},
		});

	for _ in 0..NB_RANDOM_TESTS {
		let alice_qubit = Ket::random(2);
		let shared_qubit = Ket::base(0b0, 2);
		let bob_qubit = Ket::base(0b0, 2);

		let initial_state = StateVector::from_qubits(&[&alice_qubit, &shared_qubit, &bob_qubit]);
		let backend = DenseCPUBackend;
		let program = backend.compile(&circuit);
		let final_state = program.run(&initial_state);

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
		.then(Gate::x().on(1).control(vec![0]))
		.then(Gate::x().on(2).control(vec![0]))
		// Apply h gates
		.then(Gate::h().on(0))
		.then(Gate::h().on(1))
		.then(Gate::h().on(2))
		// Error (Z gates only)
		.then(Gate::z().on(1))
		// Apply h gates again
		.then(Gate::h().on(0))
		.then(Gate::h().on(1))
		.then(Gate::h().on(2))
		// Repair
		.then(Gate::x().on(3).control(vec![0]))
		.then(Gate::x().on(3).control(vec![1]))
		.then(Gate::x().on(4).control(vec![1]))
		.then(Gate::x().on(4).control(vec![2]))
		// Catch errors
		.then_op(QuantumOperation::Measure(3))
		.then_op(QuantumOperation::Measure(4))
		// Kraus operator based on the error
		.then_op(QuantumOperation::ClassicalControl {
			look_up_table: vec![
				(
					vec![(3, 0), (4, 0)],
					Gate::from(id.kronecker_product(&id).kronecker_product(&id)).on_qubits(vec![0, 1, 2, 3, 4]),
				),
				(
					vec![(3, 0), (4, 1)],
					Gate::from(id.kronecker_product(&id).kronecker_product(&x)).on_qubits(vec![0, 1, 2, 3, 4]),
				),
				(
					vec![(3, 1), (4, 0)],
					Gate::from(x.kronecker_product(&id).kronecker_product(&id)).on_qubits(vec![0, 1, 2, 3, 4]),
				),
				(
					vec![(3, 1), (4, 1)],
					Gate::from(id.kronecker_product(&x).kronecker_product(&id)).on_qubits(vec![0, 1, 2, 3, 4]),
				),
			],
		})
		// Decode
		.then(Gate::x().on(1).control(vec![0]))
		.then(Gate::x().on(2).control(vec![0]));

	let backend = DenseCPUBackend;
	let program = backend.compile(&circuit);

	for _ in 0..NB_RANDOM_TESTS {
		let q0 = Ket::random(2);
		let q1 = Ket::base(0b0, 2);
		let q2 = Ket::base(0b0, 2);
		let q3 = Ket::base(0b0, 2);
		let q4 = Ket::base(0b0, 2);
		let initial_state = StateVector::from_qubits(&[&q0, &q1, &q2, &q3, &q4]);
		let final_state = program.run(&initial_state);
		let recovered_q0 = final_state.extract_state_of_single_qubit(0);
		dbg!(&q0);
		dbg!(&recovered_q0);
		assert!(q0.approx_eq(&recovered_q0, 1e-4));
	}
}

#[test]
fn qft_1_qubit() {
	// QFT with 1 qubit is just H
	let given_circuit = Circuit::new(1).then(Gate::h().on(0));
	let built_circuit = qft_circuit(1);
	let actual_matrix = qft_matrix(1);

	let backend = DenseCPUBackend;
	let given_circuit = backend.compile(&given_circuit);
	let built_circuit = backend.compile(&built_circuit);

	dbg!(&actual_matrix);
	dbg!(&given_circuit);
	dbg!(&built_circuit);
	dbg!(&given_circuit.as_matrix().unwrap());
	dbg!(&built_circuit.as_matrix().unwrap());

	assert!(given_circuit.as_matrix().unwrap().approx_eq(&actual_matrix, 1e-6));
	assert!(built_circuit.as_matrix().unwrap().approx_eq(&actual_matrix, 1e-6));
}

#[test]
fn qft_2_qubit() {
	let backend = DenseCPUBackend;
	// 2-qubit QFT:
	// - H on qubit 0
	// - controlled R2 (control qubit 1 -> target qubit 0)
	// - H on qubit 1
	// - swap qubit 0 and 1
	let given_circuit = Circuit::new(2)
		.then(Gate::h().on(0))
		.then({
			let angle = 2.0 * PI / 4.0; // R2
			let r2 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			Gate::from(r2).on(0).control(vec![1])
		})
		.then(Gate::h().on(1))
		.then(Gate::swap().on_qubits(vec![0, 1]));
	let given_program = backend.compile(&given_circuit);

	let reference_matrix = qft_matrix(2);
	let built_circuit = qft_circuit(2);
	let built_program = backend.compile(&built_circuit);

	dbg!(&built_circuit);
	dbg!(&given_circuit);
	assert!(built_circuit == given_circuit);

	dbg!(&reference_matrix);
	dbg!(&built_program.as_matrix().unwrap());
	dbg!(&given_program.as_matrix().unwrap());
	assert!(reference_matrix.approx_eq(&given_program.as_matrix().unwrap(), 1e-6));
	assert!(reference_matrix.approx_eq(&built_program.as_matrix().unwrap(), 1e-6));
}

#[test]
fn qft_3_qubit() {
	let backend = DenseCPUBackend;
	// 3-qubit QFT circuit:
	// - H on qubit 0
	// - controlled R2 (1 -> 0)
	// - controlled R3 (2 -> 0)
	// - H on qubit 1
	// - controlled R2 (2 -> 1)
	// - H on qubit 2
	// - swap 0 <-> 2
	let given_circuit = Circuit::new(3)
		.then(Gate::h().on(0))
		.then({
			let angle = 2.0 * PI / 4.0; // R2
			let r2 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			Gate::from(r2).on(0).control(vec![1])
		})
		.then({
			let angle = 2.0 * PI / 8.0; // R3
			let r3 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			Gate::from(r3).on(0).control(vec![2])
		})
		.then(Gate::h().on(1))
		.then({
			let angle = 2.0 * PI / 4.0; // R2 (2 -> 1)
			let r2 = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			Gate::from(r2).on(1).control(vec![2])
		})
		.then(Gate::h().on(2))
		.then(Gate::swap().on_qubits(vec![0, 2]));

	let given_program = backend.compile(&given_circuit);

	let reference_matrix = qft_matrix(3);
	let built_circuit = qft_circuit(3);
	let built_program = backend.compile(&built_circuit);

	dbg!(&built_circuit);
	dbg!(&given_circuit);
	assert!(built_circuit == given_circuit);

	dbg!(&reference_matrix);
	dbg!(&built_program.as_matrix().unwrap());
	dbg!(&given_program.as_matrix().unwrap());
	assert!(reference_matrix.approx_eq(&given_program.as_matrix().unwrap(), 1e-6));
	assert!(reference_matrix.approx_eq(&built_program.as_matrix().unwrap(), 1e-6));
}

#[test]
fn qft() {
	let frequency = 5;
	let backend = DenseCPUBackend;

	for nb_qubits in 4..=8 {
		let dimension = 2_usize.pow(nb_qubits as u32);

		// Check if circuit and matrix match
		let circuit = qft_circuit(nb_qubits);
		let circuit = backend.compile(&circuit);
		let matrix = qft_matrix(nb_qubits as u32);
		dbg!(&circuit);
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
