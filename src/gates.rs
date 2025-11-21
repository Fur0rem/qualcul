use num::Complex;

use crate::{ComplexMatrix, state::Ket};

pub struct Gate {
	op: ComplexMatrix,
}

impl Gate {
	pub fn from(op: ComplexMatrix) -> Self {
		Self { op }
	}

	pub fn as_matrix(&self) -> &ComplexMatrix {
		&self.op
	}

	pub fn x() -> Self {
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(0.0), Complex::from(1.0)],
			vec![Complex::from(1.0), Complex::from(0.0)],
		]))
	}

	pub fn z() -> Self {
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(1.0), Complex::from(0.0)],
			vec![Complex::from(0.0), Complex::from(-1.0)],
		]))
	}

	pub fn y() -> Self {
		Self::from(ComplexMatrix::from(&vec![
			vec![Complex::from(0.0), -Complex::i()],
			vec![Complex::i(), Complex::from(0.0)],
		]))
	}

	pub fn h() -> Self {
		Self::from(
			ComplexMatrix::from(&vec![
				vec![Complex::from(1.0), Complex::from(1.0)],
				vec![Complex::from(1.0), Complex::from(-1.0)],
			]) * Complex::from(1.0 / 2.0f64.sqrt()),
		)
	}

	pub fn swap() -> Self {
		let mut op = ComplexMatrix::zero(4);
		op[(0, 0)] = Complex::from(1.0);
		op[(1, 2)] = Complex::from(1.0);
		op[(2, 1)] = Complex::from(1.0);
		op[(3, 3)] = Complex::from(1.0);
		return Self::from(op);
	}

	pub fn controlled_adjacent(gate: &Self) -> Self {
		let size_side = gate.op.size_side() * 2;
		let mut op = ComplexMatrix::identity(size_side);

		for i in gate.op.size_side()..size_side {
			for j in gate.op.size_side()..size_side {
				op[(i, j)] = gate.op[(i - gate.op.size_side(), j - gate.op.size_side())];
			}
		}

		return Self::from(op);
	}

	// Controlled gate but with control and target qubits swapped
	// TODO: Verify more thoroughly the implementation with more tests, because I'm not so sure of the formula
	pub fn controlled_adjacent_reversed(gate: &Self) -> Self {
		let size_side = gate.op.size_side() * 2;
		let mut op = ComplexMatrix::identity(size_side);

		for i in 0..gate.op.size_side() {
			for j in 0..gate.op.size_side() {
				op[(i * 2 + 1, j * 2 + 1)] = gate.op[(i, j)];
			}
		}

		return Self::from(op);
	}

	// https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu/4254#4254
	pub fn controlled(gate: &Self, control_qubit: usize, target_qubit: usize, nb_qubits: usize) -> Gate {
		assert!(control_qubit != target_qubit);
		assert!(control_qubit < nb_qubits);
		assert!(target_qubit < nb_qubits);

		let projector_0 = Ket::base(0b0, 2).projector();
		let projector_1 = Ket::base(0b1, 2).projector();
		let id2 = ComplexMatrix::identity(2);

		let mut terms_when_control_is_0 = vec![&id2; nb_qubits];
		terms_when_control_is_0[control_qubit] = &projector_0;
		terms_when_control_is_0[target_qubit] = &id2;

		let mut terms_when_control_is_1 = vec![&id2; nb_qubits];
		terms_when_control_is_1[control_qubit] = &projector_1;
		terms_when_control_is_1[target_qubit] = &gate.as_matrix();

		let mut op_projected_0 = terms_when_control_is_0[0].clone();
		for term in &terms_when_control_is_0[1..] {
			op_projected_0 = op_projected_0.kronecker_product(term);
		}

		let mut op_projected_1 = terms_when_control_is_1[0].clone();
		for term in &terms_when_control_is_1[1..] {
			op_projected_1 = op_projected_1.kronecker_product(term);
		}

		return Self::from(op_projected_0 + op_projected_1);
	}
}
