use num::Complex;

use crate::ComplexMatrix;

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

	pub fn controlled(gate: &Self) -> Self {
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
	pub fn reverse_controlled(gate: &Self) -> Self {
		let size_side = gate.op.size_side() * 2;
		let mut op = ComplexMatrix::identity(size_side);

		for i in 0..gate.op.size_side() {
			for j in 0..gate.op.size_side() {
				op[(i * 2 + 1, j * 2 + 1)] = gate.op[(i, j)];
			}
		}

		return Self::from(op);
	}
}
