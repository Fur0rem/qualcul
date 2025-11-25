use num::{Complex, complex::ComplexFloat};

use crate::{ComplexMatrix, Gate, state::Ket};

#[derive(Clone, PartialEq, Default)]
pub struct StateVector(Vec<Complex<f64>>);

impl std::fmt::Debug for StateVector {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let max_bits = (self.0.len() as f64).log2().ceil() as usize;
		writeln!(f, "StateVector (")?;
		for (idx, amplitude) in self.0.iter().enumerate() {
			let binary_state = format!("{:0width$b}", idx, width = max_bits);
			writeln!(f, "    |{}>: {}+{}i", binary_state, amplitude.re(), amplitude.im())?;
		}
		write!(f, ")")?;
		return Ok(());
	}
}
impl StateVector {
	pub fn from_vec(state: Vec<Complex<f64>>) -> Self {
		StateVector(state)
	}

	pub fn from_ket(state: &Ket) -> Self {
		Self::from_vec(state.components().iter().map(|z| *z).collect())
	}

	pub fn approx_eq(&self, rhs: &StateVector, epsilon: f64) -> bool {
		if self.0.len() != rhs.0.len() {
			return false;
		}
		for i in 0..self.0.len() {
			let difference = self.0[i] - rhs.0[i];
			if difference.re() > epsilon || difference.im() > epsilon {
				return false;
			}
		}
		return true;
	}

	pub fn possible_outcomes(&self) -> Vec<(Ket, f64)> {
		let mut outcomes = Vec::new();
		for (idx, amplitude) in self.0.iter().enumerate() {
			let probability = amplitude.norm_sqr();
			if probability > 0.0 {
				outcomes.push((Ket::base(idx, self.0.len()), probability));
			}
		}
		return outcomes;
	}
}

impl std::ops::Mul<StateVector> for ComplexMatrix {
	type Output = StateVector;
	fn mul(self, rhs: StateVector) -> Self::Output {
		let mut result = StateVector(vec![Complex::from(0.0); rhs.0.len()]);
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				result.0[i] += self[(i, j)] * rhs.0[j];
			}
		}
		return result;
	}
}

impl std::ops::Index<usize> for StateVector {
	type Output = Complex<f64>;

	fn index(&self, index: usize) -> &Self::Output {
		return &self.0[index];
	}
}

impl std::ops::IndexMut<usize> for StateVector {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		return &mut self.0[index];
	}
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Circuit {
	steps: Vec<Gate>,
}

impl Circuit {
	pub fn new() -> Self {
		Self { steps: Vec::new() }
	}

	pub fn then(self, op: Gate) -> Self {
		if !self.steps.is_empty() {
			assert!(op.as_matrix().size_side() == self.steps[0].as_matrix().size_side());
		}
		let mut new_circuit = self;
		new_circuit.steps.push(op);
		return new_circuit;
	}

	pub fn as_matrix(&self) -> ComplexMatrix {
		let mut matrix = ComplexMatrix::identity(self.steps[0].as_matrix().size_side());
		for op in &self.steps {
			matrix = op.as_matrix() * &matrix;
		}
		return matrix;
	}

	// State vector implementation
	pub fn run(&self, initial_state: StateVector) -> StateVector {
		return self.as_matrix() * initial_state;
	}
}
