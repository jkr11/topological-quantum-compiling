import qiskit
from exact_synthesis.synthesis import synthesize_unitary, ExactUnitary, evaluate_gate_sequence
import numpy as np

U_matrix = np.eye(2)


def trace_norm(A: np.ndarray, B: ExactUnitary) -> float:
  return np.sum(np.linalg.svd(A - np.array(B.to_matrix().tolist(), dtype=complex), compute_uv=False))


print(synthesize_unitary(U_matrix, 10e-10))
print("Trace norm: ", trace_norm(U_matrix, evaluate_gate_sequence(synthesize_unitary(U_matrix, 10e-10))))
