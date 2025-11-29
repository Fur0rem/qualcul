use num::{Complex, Zero};

use crate::{
	ComplexMatrix, Gate, GateOperation,
	backend::{Backend, Program},
	circuit::{Circuit, StateVector},
	gates,
};

#[derive(Debug, Clone, PartialEq)]
pub struct DenseCPUBackend;

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixLookupTable {
	table: Vec<(Vec<(usize, u8)>, ComplexMatrix)>,
}

impl MatrixLookupTable {
	pub fn get(&self, bit_values: &[(usize, u8)]) -> Option<&ComplexMatrix> {
		let mut gate_to_apply = None;
		for (conditions, gate) in &self.table {
			let mut all_conditions_met = true;
			for (bit_idx, expected_value) in conditions {
				let mut found = false;
				for (known_bit_idx, known_value) in bit_values {
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
				gate_to_apply = Some(gate);
			}
		}
		return gate_to_apply;
	}

	pub fn from(look_up_table: &Vec<(Vec<(usize, u8)>, GateOperation)>, nb_qubits: usize) -> Self {
		let mut table = Vec::new();
		for (values, gate) in look_up_table {
			let matrix = as_matrix(&gate, nb_qubits);
			table.push((values.clone(), matrix));
		}
		return Self { table };
	}
}

#[derive(Debug, Clone, PartialEq)]
enum BackendOperation {
	Matrix(ComplexMatrix), // Apply complex matrix operator
	Measure(usize),        // Measure a qubit
	ClassicalControl {
		look_up_table: MatrixLookupTable, // Gates to apply based on classical bits
	},
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseCPUProgram {
	operations: Vec<BackendOperation>,
	nb_qubits: usize,
}

impl DenseCPUProgram {
	// Return matrix if compilable as a single matrix
	pub fn as_matrix(&self) -> Option<&ComplexMatrix> {
		if self.operations.len() == 1 {
			if let BackendOperation::Matrix(matrix) = &self.operations[0] {
				return Some(matrix);
			}
		}
		return None;
	}

	pub fn from_matrix(matrix: ComplexMatrix) -> Self {
		let dim = matrix.size_side();
		assert!(dim.is_power_of_two(), "matrix must be 2^n x 2^n");
		let nb_qubits: usize = dim.trailing_zeros() as usize; // log2

		Self {
			operations: vec![BackendOperation::Matrix(matrix)],
			nb_qubits,
		}
	}
}

pub fn controlled(gate: &Gate) -> Gate {
	let size_side = gate.op.size_side() * 2;
	let mut op = ComplexMatrix::identity(size_side);

	for i in gate.op.size_side()..size_side {
		for j in gate.op.size_side()..size_side {
			op[(i, j)] = gate.op[(i - gate.op.size_side(), j - gate.op.size_side())];
		}
	}

	return Gate::from(op);
}

pub fn map(gate: &Gate, mappings: &[usize]) -> Gate {
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

fn as_matrix(gate: &GateOperation, nb_qubits: usize) -> ComplexMatrix {
	// Control it as many times as needed
	let mut op = gate.op.clone();
	for _ in 0..gate.controlled_by.len() {
		op = controlled(&op);
	}

	// Map and permute the resulting matrix: first controls, then targets, then unaffected
	let mut mappings = vec![];
	for qubit in &gate.controlled_by {
		mappings.push(*qubit);
	}
	for qubit in &gate.applied_on {
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

impl Backend<DenseCPUProgram> for DenseCPUBackend {
	fn compile(&self, circuit: &Circuit) -> DenseCPUProgram {
		let nb_qubits = circuit.nb_qubits();
		let dimension = 2_usize.pow(nb_qubits as u32);

		// Fuse matrices found in a row, and for other operations just add them in order.
		let mut operations = vec![];
		let mut current_matrix = ComplexMatrix::identity(dimension);
		for operation in circuit.steps() {
			match operation {
				gates::QuantumOperation::Gate(gate) => {
					let matrix = as_matrix(&gate, nb_qubits);
					current_matrix = matrix * &current_matrix;
				}
				gates::QuantumOperation::ClassicalControl { look_up_table } => {
					operations.push(BackendOperation::Matrix(current_matrix));
					current_matrix = ComplexMatrix::identity(dimension);
					operations.push(BackendOperation::ClassicalControl {
						look_up_table: MatrixLookupTable::from(&look_up_table, nb_qubits),
					});
				}
				gates::QuantumOperation::Measure(on) => {
					operations.push(BackendOperation::Matrix(current_matrix));
					current_matrix = ComplexMatrix::identity(dimension);
					operations.push(BackendOperation::Measure(*on));
				}
			}
		}
		// Push matrix built if ended with a matrix
		if let Some(gates::QuantumOperation::Gate(_)) = circuit.steps().last() {
			operations.push(BackendOperation::Matrix(current_matrix));
		}

		return DenseCPUProgram { operations, nb_qubits };
	}
}

impl DenseCPUProgram {
	fn run_and_branch(steps: &[BackendOperation], state: &StateVector, bit_values: &[(usize, u8)], nb_qubits: usize) -> StateVector {
		// If end of program is reached, return the calculated state
		if steps.is_empty() {
			return state.clone();
		}

		match &steps[0] {
			BackendOperation::Matrix(matrix) => {
				let new_state = matrix * state;
				return Self::run_and_branch(&steps[1..], &new_state, bit_values, nb_qubits);
			}
			BackendOperation::Measure(on_qubit) => {
				// Compute the outcomes if the qubit ends up being measured as 0 or 1 and how likely each one are
				let mut outcomes = Vec::new();
				for possible_qubit_value in [0u8, 1] {
					// Loop over all possible combinations of outcomes, and keep only the ones where the desired qubit has the desired value
					// And sum their probabilities (amplitudes squared) to know how likely it is to measure it in that value.
					let mut state_if_measured = vec![Complex::zero(); state.0.len()];
					let mut probability_to_measure = 0.0;
					for possible_outcome in 0..state.0.len() {
						let qubit_value = possible_outcome >> (nb_qubits - 1 - on_qubit) & 1;
						if qubit_value as u8 == possible_qubit_value {
							state_if_measured[possible_outcome] = state[possible_outcome];
							probability_to_measure += state[possible_outcome].norm_sqr();
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
				let mut combined_outcomes = StateVector::from_vec(vec![Complex::zero(); state.0.len()]);
				let remaining_operations = &steps[1..];
				for (bit_value, probability, branched_state) in outcomes {
					let mut bit_values: Vec<_> = bit_values.iter().map(|x| *x).collect();
					bit_values.push((*on_qubit, bit_value));
					let final_state =
						Self::run_and_branch(remaining_operations, &branched_state, &bit_values, nb_qubits);
					for i in 0..final_state.0.len() {
						combined_outcomes[i] += probability * final_state[i];
					}
				}
				return combined_outcomes;
			}

			BackendOperation::ClassicalControl { look_up_table } => {
				// Find gate to apply and remove the placeholder one, then go on with the rest of the computation
				let gate_to_apply = look_up_table
					.get(bit_values)
					.expect("No matching condition found in classical control");

				let mut new_steps: Vec<_> = steps.iter().map(|x| x.clone()).collect();
				new_steps[0] = BackendOperation::Matrix(gate_to_apply.clone());

				return Self::run_and_branch(&new_steps, state, &bit_values, nb_qubits);
			}
		}
	}
}

impl Program for DenseCPUProgram {
	fn run(&self, state: &StateVector) -> StateVector {
		return Self::run_and_branch(&self.operations, state, &[], self.nb_qubits);
	}
}
