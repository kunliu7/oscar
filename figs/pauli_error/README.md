This folder contains QAOA landscape under Pauli X and Z error,
i.e. phase flip and bit flip error.

## Subfolder name explanation

```python
def get_pauli_error_noise_model(p_error: float):
    bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
    phase_flip = pauli_error([('Z', p_error), ('I', 1 - p_error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    noise_model.add_all_qubit_quantum_error(bitphase_flip, ['u1', 'u2', 'u3'])
```
- pauliXZ0.01: `get_pauli_error_noise_model(p_error: float)`
- nQ8: # qubit = 8
- p2: p = 2

Each subfolder has figs of 3 graphs.
