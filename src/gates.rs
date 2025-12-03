//! Quantum gates and operations on qubits which can be used in [circuits](crate::circuit).

use num::Complex;

use crate::ComplexMatrix;

/// A quantum gate in a circuit, represented as a [complex matrix](crate::complex_matrix).
/// The number of qubits it's applied on is the logarithm-2 of the the size of the side of the matrix, i.e. for `n` qubits, the matrix will have shape `2^n * 2^n`
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Gate {
	/// The operation associated with the gate.
	pub(crate) op: ComplexMatrix,
}

impl Gate {
	/// Creates a gate from the specified matrix.
	///
	/// # Arguments
	///
	/// * `op` - The operation of the gate to apply
	pub fn from(op: ComplexMatrix) -> Self {
		Self { op }
	}

	/// Gets the matrix of the gate.
	pub fn as_matrix(&self) -> &ComplexMatrix {
		&self.op
	}

	/// Creates a gate which does nothing on a single qubit.
	pub fn none() -> Self {
		Self::from(ComplexMatrix::identity(2))
	}

	/// Creates an X-gate on a single qubit.
	pub fn x() -> Self {
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(0.0), Complex::from(1.0)],
			vec![Complex::from(1.0), Complex::from(0.0)],
		]))
	}

	/// Creates a Z-gate on a single qubit.
	pub fn z() -> Self {
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(1.0), Complex::from(0.0)],
			vec![Complex::from(0.0), Complex::from(-1.0)],
		]))
	}

	/// Creates a Y-gate on a single qubit.
	pub fn y() -> Self {
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(0.0), -Complex::i()],
			vec![Complex::i(), Complex::from(0.0)],
		]))
	}

	/// Creates an H or Hadamard gate on a single qubit.
	pub fn h() -> Self {
		Self::from(
			ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(1.0)],
				vec![Complex::from(1.0), Complex::from(-1.0)],
			]) * Complex::from(1.0 / 2.0f64.sqrt()),
		)
	}

	/// Creates a swap gate on 2 qubits.
	pub fn swap() -> Self {
		let mut op = ComplexMatrix::zero(4);
		op[(0, 0)] = Complex::from(1.0);
		op[(1, 2)] = Complex::from(1.0);
		op[(2, 1)] = Complex::from(1.0);
		op[(3, 3)] = Complex::from(1.0);
		return Self::from(op);
	}

	/// Creates a rotation on x or Rx-gate on a single qubit.
	///
	/// # Arguments
	///
	/// * `angle` - The angle to rotate around the x-axis.
	pub fn rx(angle: f64) -> Self {
		let half_theta = angle / 2.0;
		let cos_half_theta = Complex::from(half_theta.cos());
		let minus_i_sin_half_theta = -Complex::<f64>::i() * Complex::from(half_theta.sin());

		Self::from(ComplexMatrix::from(&vec![
			vec![cos_half_theta, minus_i_sin_half_theta],
			vec![minus_i_sin_half_theta, cos_half_theta],
		]))
	}

	/// Creates a rotation on y or Ry-gate on a single qubit.
	///
	/// # Arguments
	///
	/// * `angle` - The angle to rotate around the y-axis.
	pub fn ry(angle: f64) -> Self {
		let half_theta = angle / 2.0;
		let cos_half_theta = Complex::from(half_theta.cos());
		let sin_half_theta = Complex::from(half_theta.sin());

		Self::from(ComplexMatrix::from(&vec![
			vec![cos_half_theta, -sin_half_theta],
			vec![sin_half_theta, cos_half_theta],
		]))
	}

	/// Creates a rotation on z or Rz-gate on a single qubit.
	///
	/// # Arguments
	///
	/// * `angle` - The angle to rotate around the z-axis.
	pub fn rz(angle: f64) -> Self {
		let half_theta = angle / 2.0;
		let exp_minus_i_half_theta = Complex::<f64>::from_polar(1.0, -half_theta);
		let exp_i_half_theta = Complex::<f64>::from_polar(1.0, half_theta);

		Self::from(ComplexMatrix::from(&vec![
			vec![exp_minus_i_half_theta, Complex::from(0.0)],
			vec![Complex::from(0.0), exp_i_half_theta],
		]))
	}

	/// Creates a phase shift gate on a single qubit.
	///
	/// # Arguments
	///
	/// * `angle` - The angle with which to shift the phase.
	pub fn phase_shift(angle: f64) -> Self {
		let exp_i_phi: Complex<f64> = Complex::<f64>::from_polar(1.0, angle);

		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(1.0), Complex::from(0.0)],
			vec![Complex::from(0.0), exp_i_phi],
		]))
	}

	/// Creates an S-gate on a single qubit.
	pub fn s() -> Self {
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(1.0), Complex::from(0.0)],
			vec![Complex::from(0.0), Complex::i()],
		]))
	}

	/// Creates a T-gate on a single qubit.
	pub fn t() -> Self {
		let exp_i_pi_over_4: Complex<f64> = Complex::<f64>::from_polar(1.0, std::f64::consts::PI / 4.0);
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(1.0), Complex::from(0.0)],
			vec![Complex::from(0.0), exp_i_pi_over_4],
		]))
	}

	/// Creates a phase gate on a single qubit.
	///
	/// # Arguments
	///
	/// * `angle` - The angle with which to phase.
	pub fn phase(angle: f64) -> Self {
		let exp_i_alpha: Complex<f64> = Complex::<f64>::from_polar(1.0, angle);
		Self::from(ComplexMatrix::from(&vec![
			vec![exp_i_alpha, Complex::from(0.0)],
			vec![Complex::from(0.0), exp_i_alpha],
		]))
	}
}

/// A gate that is applied on qubits in a quantum circuit.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AppliedGate {
	/// The gate to apply.
	pub(crate) op:            Gate,
	/// The qubits it's applied on.
	pub(crate) applied_on:    Vec<usize>,
	/// The control qubits.
	pub(crate) controlled_by: Vec<usize>,
}

impl Gate {
	/// Applies the gate on the specified qubit.
	///
	/// # Arguments
	///
	/// * `qubit` - The index of the qubit to apply the gate on.
	pub fn on(self, qubit: usize) -> AppliedGate {
		AppliedGate {
			op:            self,
			applied_on:    vec![qubit],
			controlled_by: vec![],
		}
	}

	/// Applies the gate on the specified qubits.
	///
	/// # Arguments
	///
	/// * `qubits` - The indices of the qubits to apply the gate on.
	pub fn on_qubits(self, qubits: Vec<usize>) -> AppliedGate {
		AppliedGate {
			op:            self,
			applied_on:    qubits,
			controlled_by: vec![],
		}
	}
}

impl AppliedGate {
	/// Controls the gate with the given qubits.
	///
	/// # Arguments
	///
	/// * `by` - The indices of the control qubits.
	pub fn control(self, by: Vec<usize>) -> Self {
		Self {
			op:            self.op,
			applied_on:    self.applied_on,
			controlled_by: by,
		}
	}

	/// Get the operation of the gate.
	pub fn op(&self) -> &Gate {
		&self.op
	}

	/// Gets the qubits the gate is applied on.
	pub fn applied_on(&self) -> &Vec<usize> {
		&self.applied_on
	}

	/// Gets the qubits the gate is controlled by.
	pub fn controlled_by(&self) -> &Vec<usize> {
		&self.controlled_by
	}
}

/// Computes the controlled version of the gate
pub(crate) fn controlled(gate: &Gate) -> Gate {
	let size_side = gate.op.size_side() * 2;
	let mut op = ComplexMatrix::identity(size_side);

	for i in gate.op.size_side()..size_side {
		for j in gate.op.size_side()..size_side {
			op[(i, j)] = gate.op[(i - gate.op.size_side(), j - gate.op.size_side())];
		}
	}

	return Gate::from(op);
}

/// Creates the gate where the indices are mapped.
/// Each gate assumes its applied on the qubits from `0` to `n`, in order.
/// So this turns the matrix into the one where the qubits its applied on in specified in `mappings`.
///
/// For example, the CX (controlled-X) gate assumes the X is applied on qubit 1 and controlled by qubit 0.
/// With mappings [3, 5], the gate will now apply the X on qubit 5, controlled by qubit 0, assuming a 6-qubits circuit.
pub(crate) fn map(gate: &Gate, mappings: &[usize]) -> Gate {
	assert!(!mappings.is_empty());

	// Turn the matrix into a full one by filling identity on the unmapped qubits
	let id2 = ComplexMatrix::identity(2);
	let mut gate = gate.op.clone();
	for _ in 0..mappings.len() - (gate.size_side() as f64).log2() as usize {
		gate = gate.kronecker_product(&id2);
	}

	// Permute the rows and columns according to the mappings
	let nb_qubits = mappings.len();
	let size_side = 1 << nb_qubits;
	let mut full_op = ComplexMatrix::identity(size_side);
	for i in 0..size_side {
		for j in 0..size_side {
			let mut row = 0;
			let mut col = 0;
			for (k, mapping) in mappings.iter().enumerate() {
				let bit_i = (i >> (nb_qubits - 1 - k)) & 1;
				let bit_j = (j >> (nb_qubits - 1 - k)) & 1;
				row |= bit_i << (mappings.len() - 1 - mapping);
				col |= bit_j << (mappings.len() - 1 - mapping);
			}
			full_op[(row, col)] = gate[(i, j)];
		}
	}
	return Gate::from(full_op);
}

impl AppliedGate {
	/// Computes the matrix representation of the gate operation.
	///
	/// # Arguments
	///
	/// * `nb_qubits` - The number of qubits in the circuit.
	pub fn into_matrix(&self, nb_qubits: usize) -> ComplexMatrix {
		// Control it as many times as needed
		let mut op = self.op.clone();
		for _ in 0..self.controlled_by.len() {
			op = controlled(&op);
		}

		// Map and permute the resulting matrix: first controls, then targets, then unaffected
		let mut mappings = vec![];
		for qubit in &self.controlled_by {
			assert!(*qubit < nb_qubits);
			mappings.push(*qubit);
		}
		for qubit in &self.applied_on {
			assert!(*qubit < nb_qubits);
			mappings.push(*qubit);
		}
		for q in 0..nb_qubits {
			if !mappings.contains(&q) {
				mappings.push(q);
			}
		}
		op = map(&op, &mappings);

		return op.as_matrix().clone();
	}
}

/// A quantum operation on qubits in a circuit.
/// Can be a gate, measure, or a gate influenced by classical bits.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumOperation {
	/// Quantum gate
	Gate(AppliedGate),
	/// Measure a qubit
	Measure(usize),
	/// A gate influenced by classical bits.
	ClassicalControl {
		/// Gates to apply based on classical bits
		look_up_table: Vec<(Vec<(usize, u8)>, AppliedGate)>,
	},
}
