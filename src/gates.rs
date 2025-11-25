use num::Complex;

use crate::ComplexMatrix;

#[derive(Debug, Clone, PartialEq, Default)]
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

	pub fn map(gate: &Self, mappings: &[usize]) -> Self {
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
		return Self::from(full_op);
	}
}
