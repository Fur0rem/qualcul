use std::vec;

use num::{
	Complex, Zero,
	complex::{Complex64, ComplexFloat},
};

use crate::{ComplexMatrix, GateOperation, QuantumOperation, state::Ket};

#[derive(Clone, PartialEq, Default)]
pub struct StateVector(pub(crate) Vec<Complex<f64>>);

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

	pub fn from_qubits(qubit_states: &[&Ket]) -> Self {
		let mut result = qubit_states[0].clone();
		for qubit_state in &qubit_states[1..] {
			result = result.kronecker_product(qubit_state);
		}
		Self::from_vec(result.components().to_vec())
	}

	pub fn approx_eq(&self, rhs: &StateVector, epsilon: f64) -> bool {
		if self.0.len() != rhs.0.len() {
			return false;
		}
		for i in 0..self.0.len() {
			let difference = self.0[i] - rhs.0[i];
			if difference.re().abs() > epsilon || difference.im().abs() > epsilon {
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

	pub fn extract_state_of_single_qubit(&self, qubit: usize) -> Ket {
		let dim = self.0.len();
		let mut state0 = Complex64::zero();
		let mut state1 = Complex64::zero();
		for i in 0..dim {
			let qubit_value = (i >> (dim.trailing_zeros() - 1 - qubit as u32)) & 1;
			if qubit_value == 0 {
				state0 += self.0[i];
			} else {
				state1 += self.0[i];
			}
		}
		let norm = (state0.norm_sqr() + state1.norm_sqr()).sqrt();
		if norm == 0.0 {
			return Ket::base(0, 2);
		}
		return Ket::from_components(vec![state0 / Complex::from(norm), state1 / Complex::from(norm)]);
	}

	pub fn most_likely_outcome(&self) -> usize {
		self.components()
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
			.map(|(i, _)| i)
			.unwrap()
	}

	pub fn components(&self) -> &[Complex<f64>] {
		&self.0
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

impl std::ops::Mul<&StateVector> for ComplexMatrix {
	type Output = StateVector;
	fn mul(self, rhs: &StateVector) -> Self::Output {
		let mut result = StateVector(vec![Complex::from(0.0); rhs.0.len()]);
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				result.0[i] += self[(i, j)] * rhs.0[j];
			}
		}
		return result;
	}
}

impl std::ops::Mul<&StateVector> for &ComplexMatrix {
	type Output = StateVector;
	fn mul(self, rhs: &StateVector) -> Self::Output {
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
	steps: Vec<QuantumOperation>,
	nb_qubits: usize,
}

impl Circuit {
	pub fn new(nb_qubits: usize) -> Self {
		Self {
			steps: Vec::new(),
			nb_qubits,
		}
	}

	pub fn then(self, gate: GateOperation) -> Self {
		let mut new_circuit = self;
		new_circuit.steps.push(QuantumOperation::Gate(gate));
		return new_circuit;
	}

	pub fn then_op(self, op: QuantumOperation) -> Self {
		let mut new_circuit = self;
		new_circuit.steps.push(op);
		return new_circuit;
	}

	pub fn steps(&self) -> &[QuantumOperation] {
		&self.steps
	}

	pub fn nb_qubits(&self) -> usize {
		self.nb_qubits
	}
}
