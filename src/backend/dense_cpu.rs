//! Compile and run quantum programs using a simulator with dense linear algebra algorithms on the CPU.

use num::{Complex, Zero};

use crate::{
	AppliedGate, ComplexMatrix,
	backend::{Backend, BackendOperation, LookupTable, Program},
	circuit::Circuit,
	gates,
	state::StateVector,
};

/// Backend to compile programs into [DenseCPUProgram]'s
#[derive(Debug, Clone, PartialEq)]
pub struct DenseCPUBackend;

/// A program that runs a quantum algorithm with dense linear algebra calculations on the CPU.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseCPUProgram {
	/// Operations to apply in order to the state vector
	operations: Vec<BackendOperation<ComplexMatrix>>,
	/// The number of qubits the program works on.
	nb_qubits:  usize,
}

impl DenseCPUProgram {
	/// Gets the matrix representation of the program, if possible.
	/// It's not possible if there is a classical control or measure operation.
	pub fn as_matrix(&self) -> Option<&ComplexMatrix> {
		if self.operations.len() == 1 {
			if let BackendOperation::Matrix(matrix) = &self.operations[0] {
				return Some(matrix);
			}
		}
		return None;
	}

	/// Creates a program from a matrix representing the circuit.
	///
	/// # Panics
	///
	/// Panics if the matrix doesn't have a side size of a power of 2, it should have shape [2^n, 2^n].
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

impl LookupTable<ComplexMatrix> {
	/// Creates a look-up table of matrices that can be ran in the simulation.
	pub fn from(look_up_table: &Vec<(Vec<(usize, u8)>, AppliedGate)>, nb_qubits: usize) -> Self {
		let mut table = Vec::new();
		for (values, gate) in look_up_table {
			let matrix = gate.into_matrix(nb_qubits);
			table.push((values.clone(), matrix));
		}
		return Self { table };
	}
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
					let matrix = gate.into_matrix(nb_qubits);
					current_matrix = matrix * &current_matrix;
				}
				gates::QuantumOperation::ClassicalControl { look_up_table } => {
					operations.push(BackendOperation::Matrix(current_matrix));
					current_matrix = ComplexMatrix::identity(dimension);
					operations.push(BackendOperation::ClassicalControl {
						look_up_table: LookupTable::<ComplexMatrix>::from(&look_up_table, nb_qubits),
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
	/// Runs the simulation, branches into sub-simulations in cases of measure or classical control operations.
	///
	/// # Arguments
	///
	/// * `steps` - The remaining operations to be simulated.
	/// * `state` - The current state of the input state vector.
	/// * `bit_values` - The measured value (0/1) of each qubit in the case where they got measured.
	/// * `nb_qubits` - The number of qubits the program runs on.
	fn run_and_branch(
		steps: &[BackendOperation<ComplexMatrix>], state: &StateVector, bit_values: &[(usize, u8)], nb_qubits: usize,
	) -> StateVector {
		// If end of program is reached, return the calculated state
		if steps.is_empty() {
			return state.clone();
		}

		match &steps[0] {
			BackendOperation::Matrix(matrix) => {
				// Apply the matrix to the state vector and move on to the next steps.
				let new_state = matrix * state;
				return Self::run_and_branch(&steps[1..], &new_state, bit_values, nb_qubits);
			}
			BackendOperation::Measure(on_qubit) => {
				// Compute the outcomes if the qubit ends up being measured as 0 or 1 and how likely each one are.
				let mut outcomes = Vec::new();
				for possible_qubit_value in [0u8, 1] {
					// Loop over all possible combinations of outcomes, and keep only the ones where the desired qubit has the desired value.
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

					// If it can happen, normalize the state and add it to the possible outcomes.
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

				// Run simulations on the rest of the circuit assuming either state, and aggregate the results at the end.
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
				// Find gate to apply and remove the placeholder one, then go on with the rest of the computation.
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
	fn run(&mut self, state: &StateVector) -> StateVector {
		return Self::run_and_branch(&self.operations, state, &[], self.nb_qubits);
	}
}
