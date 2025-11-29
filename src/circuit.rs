use std::vec;

use num::{
	Complex, Zero,
	complex::{Complex64, ComplexFloat},
};

use crate::{ComplexMatrix, Gate, Operation, state::Ket};

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
	steps: Vec<Operation>,
	nb_qubits: usize,
}

impl Circuit {
	pub fn new(nb_qubits: usize) -> Self {
		Self {
			steps: Vec::new(),
			nb_qubits,
		}
	}

	pub fn then(self, gate: Gate) -> Self {
		let mut new_circuit = self;
		new_circuit.steps.push(Operation::Gate(gate));
		return new_circuit;
	}

	pub fn then_op(self, op: Operation) -> Self {
		let mut new_circuit = self;
		new_circuit.steps.push(op);
		return new_circuit;
	}

	pub fn as_matrix(&self) -> Option<ComplexMatrix> {
		let mut matrices = Vec::new();
		for op in &self.steps {
			match op {
				Operation::Gate(gate) => {
					matrices.push(gate.as_matrix());
				}
				_ => return None,
			}
		}
		let mut matrix = ComplexMatrix::identity(matrices[0].size_side());
		for op in matrices {
			matrix = op * &matrix;
		}
		return Some(matrix);
	}

	fn run_only_gates(state: &StateVector, operations: &[Gate], nb_qubits: usize) -> StateVector {
		let mut operation_matrix = ComplexMatrix::identity(2usize.pow(nb_qubits as u32));
		for op in operations {
			operation_matrix = op.as_matrix() * &operation_matrix;
		}
		return operation_matrix * state;
	}

	// State vector implementation
	pub fn run_and_branch(
		steps: &[Operation], initial_state: &StateVector, bit_values: &[(usize, u8)], nb_qubits: usize,
	) -> StateVector {
		// Find how much you can simulate without needing measurement nor classical interactions
		let mut gate_operations = Vec::new();
		let mut index_of_first_non_gate = 0;
		for (idx, op) in steps.iter().enumerate() {
			match op {
				Operation::Gate(gate) => {
					gate_operations.push(gate.clone());
				}
				_ => {
					index_of_first_non_gate = idx;
					break;
				}
			}
		}

		// The circuit only has gates
		if gate_operations.len() == steps.len() {
			return Self::run_only_gates(initial_state, &gate_operations, nb_qubits);
		}

		// If it runs into a measure operation, compute what would happen if 0 is measured, then if 1 is measured, and do a weighted sum of the branched states
		if let Operation::Measure(on_qubit) = steps[index_of_first_non_gate] {
			let pre_measure_state = Self::run_only_gates(initial_state, &gate_operations, nb_qubits);

			// Compute the outcomes if the qubit ends up being measured as 0 or 1 and how likely each one are
			let mut outcomes = Vec::new();
			for possible_qubit_value in [0u8, 1] {
				// Loop over all possible combinations of outcomes, and keep only the ones where the desired qubit has the desired value
				// And sum their probabilities (amplitudes squared) to know how likely it is to measure it in that value.
				let mut state_if_measured = vec![Complex::zero(); pre_measure_state.0.len()];
				let mut probability_to_measure = 0.0;
				for possible_outcome in 0..pre_measure_state.0.len() {
					let qubit_value = possible_outcome >> (nb_qubits - 1 - on_qubit) & 1;
					if qubit_value as u8 == possible_qubit_value {
						state_if_measured[possible_outcome] = pre_measure_state[possible_outcome];
						probability_to_measure += pre_measure_state[possible_outcome].norm_sqr();
					}
				}

				// If it can happen, normalize the state and add it to the possible outcomes
				if probability_to_measure > 0.0 {
					for amplitude in state_if_measured.iter_mut() {
						*amplitude /= Complex::from(probability_to_measure);
					}
					outcomes.push((
						possible_qubit_value,
						probability_to_measure,
						StateVector::from_vec(state_if_measured),
					));
				}
			}

			// Run simulations on the rest of the circuit assuming either state, and aggregate the results at the end
			let mut combined_outcomes = StateVector::from_vec(vec![Complex::zero(); pre_measure_state.0.len()]);
			let remaining_operations = &steps[(index_of_first_non_gate + 1)..];
			for (bit_value, probability, middle_state) in outcomes {
				let mut bit_values: Vec<_> = bit_values.iter().map(|x| *x).collect();
				bit_values.push((on_qubit, bit_value));
				let final_state = Self::run_and_branch(remaining_operations, &middle_state, &bit_values, nb_qubits);
				for i in 0..final_state.0.len() {
					combined_outcomes[i] += probability * final_state[i];
				}
			}
			return combined_outcomes;
		}

		// If it runs into a classical control, switch the placeholder gate by the actual gate indicated by the bits, and run the rest
		if let Operation::ClassicalControl { look_up_table } = &steps[index_of_first_non_gate] {
			let bit_values: Vec<_> = bit_values.iter().map(|x| *x).collect();
			let mut gate_to_apply = None;
			for (conditions, gate) in look_up_table {
				let mut all_conditions_met = true;
				for (bit_idx, expected_value) in conditions {
					let mut found = false;
					for (known_bit_idx, known_value) in &bit_values {
						if bit_idx == known_bit_idx {
							if expected_value != known_value {
								all_conditions_met = false;
							}
							found = true;
							break;
						}
					}
					if !found {
						all_conditions_met = false;
					}
				}
				if all_conditions_met {
					gate_to_apply = Some(gate.clone());
				}
			}
			let gate_to_apply = gate_to_apply.expect("No matching condition found in classical control");

			let mut new_steps: Vec<_> = steps.iter().map(|x| x.clone()).collect();
			new_steps[index_of_first_non_gate] = Operation::Gate(gate_to_apply.clone());

			return Self::run_and_branch(&new_steps, initial_state, &bit_values, nb_qubits);
		}

		panic!("Operation was neither Gate nor Measure nor ClassicalControl");
	}

	pub fn run(&self, initial_state: &StateVector) -> StateVector {
		return Self::run_and_branch(&self.steps, initial_state, &[], self.nb_qubits);
	}

	pub fn from_matrix(matrix: ComplexMatrix) -> Self {
		let dimension: usize = matrix.size_side();
		assert!(dimension.is_power_of_two(), "matrix must be 2^n x 2^n");
		let nb_qubits = dimension.trailing_zeros() as usize;

		return Self {
			steps: vec![Operation::Gate(Gate::from(matrix))],
			nb_qubits,
		};
	}
}
