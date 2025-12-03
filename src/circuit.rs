//! Create circuits to run on a (simulated) quantum computer.
//!
//! The API uses a builder pattern to be able to chain gates more easily.

use crate::{AppliedGate, QuantumOperation};

/// A quantum circuit
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Circuit {
	/// Each operation of the circuit that needs to be executed in order.
	steps:     Vec<QuantumOperation>,
	/// The number of qubits the circuit works on.
	nb_qubits: usize,
}

impl Circuit {
	/// Creates a new, empty circuit with `nb_qubits` qubits.
	pub fn new(nb_qubits: usize) -> Self {
		Self {
			steps: Vec::new(),
			nb_qubits,
		}
	}

	/// Puts a gate as the next step in the circuit.
	pub fn then(self, gate: AppliedGate) -> Self {
		let mut new_circuit = self;
		new_circuit.steps.push(QuantumOperation::Gate(gate));
		return new_circuit;
	}

	/// Puts an operation as the next step in the circuit.
	pub fn then_op(self, op: QuantumOperation) -> Self {
		let mut new_circuit = self;
		new_circuit.steps.push(op);
		return new_circuit;
	}

	/// Gets the different steps of the circuit.
	pub fn steps(&self) -> &[QuantumOperation] {
		&self.steps
	}

	/// Gets the number of qubits inside the circuit
	pub fn nb_qubits(&self) -> usize {
		self.nb_qubits
	}
}
