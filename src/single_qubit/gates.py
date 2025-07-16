import numpy as np


class Gates(object):
  """
  Collection of common single qubit quantum gates.
  """

  # Pauli matrices
  X = np.array([[0.0, 1.0], [1.0, 0.0]])
  Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
  Z = np.array([[1.0, 0.0], [0.0, -1.0]])
  # Hadamard gate
  H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
  # S gate
  S = np.array([[1.0, 0.0], [0.0, 1.0j]])
  # T gate
  T = np.array([[1.0, 0.0], [0.0, np.sqrt(1j)]])

  @classmethod
  def R(θ):
    """
    Z-rotation gate used in the Fourier transform circuit.
    """
    return np.array([[1.0, 0.0], [0.0, np.exp(2 * np.pi * 1j * θ)]])

  @classmethod
  def Rx(cls, θ):
    """
    Rx-rotation gate.
    """
    c = np.cos(0.5 * θ)
    s = np.sin(0.5 * θ)
    return np.array([[c, -1j * s], [-1j * s, c]])

  @classmethod
  def Ry(cls, θ):
    """
    Ry-rotation gate.
    """
    c = np.cos(0.5 * θ)
    s = np.sin(0.5 * θ)
    return np.array([[c, -s], [s, c]])

  @classmethod
  def Rz(cls, θ):
    """
    Rz-rotation gate.
    """
    z = np.exp(0.5j * θ)
    return np.array([[z.conjugate(), 0], [0, z]])
