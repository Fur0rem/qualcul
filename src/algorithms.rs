use std::f64::consts::PI;

use num::{Complex, One};

use crate::{ComplexMatrix, Gate, circuit::Circuit};

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

pub fn qft_circuit(nb_qubits: usize) -> Circuit {
	let mut circuit = Circuit::new(nb_qubits);

	for target in 0..nb_qubits {
		// Apply H on the target qubit:
		let mut map_h: Vec<usize> = (0..nb_qubits).collect();
		map_h.swap(0, target);
		circuit = circuit.then(Gate::map(&Gate::h(), &map_h));

		// Apply controlled rotations from more significant qubits (control > target)
		for control in (target + 1)..nb_qubits {
			let distance = control - target;
			let angle = 2.0 * PI / 2f64.powi((distance + 1) as i32);
			let r_k = ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(0.0)],
				vec![Complex::from(0.0), Complex::from_polar(1.0, angle)],
			]);
			let phase_gate = Gate::from(r_k);

			// Build full mapping where first entry is control, second is target, then the rest untouched.
			let mut mapping = Vec::with_capacity(nb_qubits);
			mapping.push(control);
			mapping.push(target);
			for q in 0..nb_qubits {
				if q != control && q != target {
					mapping.push(q);
				}
			}

			circuit = circuit.then(Gate::map(&Gate::controlled(&phase_gate), &mapping));
		}
	}

	// Reverse qubit order by swapping pairs (i <-> n-1-i)
	for i in 0..(nb_qubits / 2) {
		let a = i;
		let b = nb_qubits - 1 - i;
		let mut mapping = Vec::with_capacity(nb_qubits);
		mapping.push(a);
		mapping.push(b);
		for q in 0..nb_qubits {
			if q != a && q != b {
				mapping.push(q);
			}
		}
		circuit = circuit.then(Gate::map(&Gate::swap(), &mapping));
	}

	circuit
}
