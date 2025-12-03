//! Some common quantum algorithms.

use std::f64::consts::PI;

use num::{Complex, One};

use crate::{ComplexMatrix, Gate, circuit::Circuit};

/// Creates a matrix representing the Quantum Fourier Transform algorithm.
///
/// # Arguments
///
/// * `nb_qubits` - The number of qubits the algorithm will run on.
pub fn qft_matrix(nb_qubits: u32) -> ComplexMatrix {
	let dimension = 2_usize.pow(nb_qubits);
	let omega = Complex::from_polar(1.0, 2.0 * PI / (dimension as f64));
	let mut matrix = ComplexMatrix::zero(dimension);

	for i in 0..dimension {
		for j in 0..dimension {
			if i == 0 || j == 0 {
				matrix[(i, j)] = Complex::one();
			}
			let exponent = (i * j) as f64;
			matrix[(i, j)] = omega.powf(exponent) * (1.0 / (dimension as f64).sqrt());
		}
	}

	return matrix;
}

/// Creates a circuit of the Quantum Fourier Transform algorithm.
///
/// # Arguments
///
/// * `nb_qubits` - The number of qubits the algorithm will run on.
pub fn qft_circuit(nb_qubits: usize) -> Circuit {
	let mut circuit = Circuit::new(nb_qubits);

	for target in 0..nb_qubits {
		// Apply H on the target qubit:
		circuit = circuit.then(Gate::h().on(target));

		// Apply controlled rotations from more significant qubits (control > target)
		for control in (target + 1)..nb_qubits {
			let distance = control - target;
			let angle = 2.0 * PI / 2f64.powi((distance + 1) as i32);
			let r_k = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			let phase_gate = Gate::from(r_k);

			circuit = circuit.then(phase_gate.on(target).control(vec![control]));
		}
	}

	// Reverse qubit order by swapping pairs (i <-> n-1-i)
	for i in 0..(nb_qubits / 2) {
		let mirror = nb_qubits - 1 - i;
		circuit = circuit.then(Gate::swap().on_qubits(vec![i, mirror]));
	}

	return circuit;
}
