# An algorithm for exact synthesis of fibonacci anyons

For now this implements https://arxiv.org/abs/1310.4150. 

## Usage

```python 
from exact_synthesis.synthesis import (
  synthesize_z_rotation,
  exact_synthesize,
)
from exact_synthesis.util import d_z, d_trace, rz_mp

mpmath.mp.dps = 100  # make a choice here

phi = mpmath.rand() # Angle for the z rotation
epsilon = mpmath.mpf("1e-8") # total error bound of approximation


print("Phi:", phi)
print("Epsilon:", epsilon)


approx = synthesize_z_rotation(phi, epsilon)

# Here we show that the d_z and the d_trace expressions are equivalent
dz_error = d_z(approx, phi)

trace_error = d_trace(raw, rz_mp(phi))

print(approx.to_matrix())

print("Z-error:", dz_error)
print("Trace distance:", trace_error)
print("Pass epsilon:", raw_dz <= epsilon)
gates = exact_synthesize(raw) # Convert to a list of basis gates (not braids yet)
print(gates)
print(len(gates)) # Scales with log(1/epsilon)
```