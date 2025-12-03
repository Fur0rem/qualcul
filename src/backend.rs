//! Provides traits for compiling and running quantum programs.

use crate::{circuit::Circuit, state::StateVector};

/// A backend turns a circuit into a [program](Program) that can be ran.
pub trait Backend<P: Program> {
	fn compile(&self, circuit: &Circuit) -> P;
}

/// A program takes a [state](StateVector) as input and computes the output after applying the circuit to it.
pub trait Program {
	fn run(&mut self, state: &StateVector) -> StateVector;
}

pub mod dense_cpu;
pub mod dense_gpu;

/// A look-up table to find the operator to apply based on the values on classical qubits.
/// Used in ClassicalControl scenarios.
#[derive(Debug, Clone, PartialEq)]
pub struct LookupTable<M> {
	table: Vec<(Vec<(usize, u8)>, M)>,
}

impl<M> LookupTable<M> {
	/// Finds the corresponding operation to apply based on the values of the classical bits, as well as a unique identifier.
	/// Returns None if it wasn't found.
	///
	/// # Arguments
	///
	/// * `bit_values` - An array of pairs (bit/qubit index, value it has (0/1))
	pub fn get_enumerate(&self, bit_values: &[(usize, u8)]) -> Option<(usize, &M)> {
		let mut gate_to_apply = None;
		for (i, (conditions, gate)) in self.table.iter().enumerate() {
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
				gate_to_apply = Some((i, gate));
				break;
			}
		}
		return gate_to_apply;
	}

	/// Finds the corresponding operation to apply based on the values of the classical bits.
	/// Returns None if it wasn't found.
	///
	/// # Arguments
	///
	/// * `bit_values` - An array of pairs (bit/qubit index, value it has (0/1))
	pub fn get(&self, bit_values: &[(usize, u8)]) -> Option<&M> {
		match self.get_enumerate(bit_values) {
			Some((_, gate)) => Some(gate),
			None => None,
		}
	}
}

/// An operation that can be used in the backend where `M` is the fundamental operation (likely some kind of matrix)
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BackendOperation<M> {
	/// Apply complex matrix operator
	Matrix(M),
	/// Measure a qubit
	Measure(usize),
	/// A gate influenced by classical bits.
	ClassicalControl {
		look_up_table: LookupTable<M>, // Gates to apply based on classical bits
	},
}
