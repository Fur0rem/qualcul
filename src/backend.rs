use crate::circuit::{Circuit, StateVector};

pub trait Backend<P: Program> {
	fn compile(&self, circuit: &Circuit) -> P;
}

pub trait Program {
	fn run(&self, state: &StateVector) -> StateVector;
}

pub mod dense_cpu;
pub mod dense_gpu;
