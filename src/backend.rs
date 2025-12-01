use crate::circuit::{Circuit, StateVector};

pub trait Backend<P: Program> {
	fn compile(&self, circuit: &Circuit) -> P;
}

pub trait Program {
	fn run(&mut self, state: &StateVector) -> StateVector;
}

pub mod dense_cpu;
pub mod dense_gpu;

#[derive(Debug, Clone, PartialEq)]
pub struct LookupTable<M> {
	table: Vec<(Vec<(usize, u8)>, M)>,
}

impl<M> LookupTable<M> {
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

	pub fn get(&self, bit_values: &[(usize, u8)]) -> Option<&M> {
		match self.get_enumerate(bit_values) {
			Some((_, gate)) => Some(gate),
			None => None,
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BackendOperation<M> {
	Matrix(M),      // Apply complex matrix operator
	Measure(usize), // Measure a qubit
	ClassicalControl {
		look_up_table: LookupTable<M>, // Gates to apply based on classical bits
	},
}
