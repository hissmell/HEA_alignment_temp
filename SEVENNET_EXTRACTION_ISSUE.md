# SevenNet Latent Vector Extraction Issue

## Problem Summary
We need to extract per-atom latent vectors from SevenNet models for representation alignment analysis, but TorchScript compilation prevents standard extraction methods.

## Background

### Project Context
- **Goal**: Extract MLIP latent representations for uncertainty estimation research
- **Dataset**: 25Cao High Entropy Alloy structures
- **Purpose**: Compare physics-inspired representations with MLIP embeddings

### Target Layer
```
[Atomic Structure Input]
        ↓
[5 Equivariant Gate Layers]
        ↓
reduce_input_to_hidden (IrrepsLinear)  ← OUTPUT = latent vectors we need
        ↓
reduce_hidden_to_energy (IrrepsLinear) ← Energy prediction head
        ↓
[Total Energy Output]
```

**Target**: Extract the OUTPUT of `reduce_input_to_hidden` layer
- Expected shape: `(n_atoms, hidden_dim)`
- Example: 37 atoms → `(37, 64)` tensor

## Technical Issue

### Core Problem
SevenNet uses TorchScript-compiled modules that prevent standard PyTorch hook registration.

### Error Details
```python
# Attempting to register hook
model.reduce_input_to_hidden.register_forward_hook(capture_hook)

# Results in:
RuntimeError: register_forward_hook is not supported on ScriptModules
```

### Module Structure
```
reduce_input_to_hidden (IrrepsLinear)
├── linear (Linear)
│   ├── weight: torch.Size([8192])
│   └── _compiled_main: RecursiveScriptModule  ← TorchScript compiled
```

## Failed Approaches

### 1. Forward Hooks
```python
# Standard PyTorch approach - FAILS
def capture_hook(module, input, output):
    captured = output.detach().cpu().numpy()

handle = model.reduce_input_to_hidden.register_forward_hook(capture_hook)
# Error: ScriptModules don't support hooks
```

### 2. Hook on Parent Module
```python
# Hook registers but captures nothing
handle = model.reduce_input_to_hidden.register_forward_hook(capture_hook)
energy = atoms.get_potential_energy()
# captured remains empty - computation happens in compiled submodule
```

### 3. Monkey Patching
```python
# Override forward method - FAILS
original_forward = model.reduce_input_to_hidden.forward

def patched_forward(x):
    output = original_forward(x)
    # Capture output here
    return output

model.reduce_input_to_hidden.forward = patched_forward
# No capture - possibly due to JIT compilation
```

## Environment

### Software Versions
- **SevenNet**: `sevenn` package
- **Model**: 7net-0 (default pretrained)
- **PyTorch**: with TorchScript/JIT compilation
- **Python**: 3.10
- **Environment**: `conda activate fairchem`

### Installation
```bash
conda install -c conda-forge sevenn
```

### Code to Reproduce
```python
from sevenn.calculator import SevenNetCalculator
from ase import Atoms

# Load model
calc = SevenNetCalculator("7net-0", device='cpu')
model = calc.model

# Create test structure
atoms = Atoms('Pt4O',
              positions=[[0,0,0], [2,0,0], [0,2,0], [2,2,0], [1,1,2]],
              cell=[10,10,10], pbc=True)
atoms.calc = calc

# Try to hook reduce_input_to_hidden
try:
    def hook(module, input, output):
        print(f"Captured: {output.shape}")

    handle = model.reduce_input_to_hidden.register_forward_hook(hook)
except RuntimeError as e:
    print(f"Error: {e}")  # ScriptModule error

# Even if hook registers on parent, nothing captured
energy = atoms.get_potential_energy()
```

## Comparison with Other MLIPs

| MLIP | Extraction Method | Status |
|------|------------------|--------|
| EquiformerV2 | Forward hooks | ✅ Works |
| MACE | Forward hooks | ✅ Works |
| UMA | Forward hooks | ✅ Works |
| Orb | Forward hooks | ✅ Works |
| **SevenNet** | **Forward hooks** | **❌ Blocked by TorchScript** |

## Question for Help

**How can we extract intermediate layer outputs from TorchScript-compiled SevenNet models?**

Specifically:
1. Is there a way to access `reduce_input_to_hidden` output despite TorchScript compilation?
2. Can we modify SevenNet to expose intermediate representations?
3. Is there an official SevenNet API for feature extraction?

## Alternative Solutions Needed

If direct extraction is impossible, we need:
1. Method to trace/export the model with intermediate outputs
2. Way to modify SevenNet source to add extraction capability
3. Official support from SevenNet developers

## Contact
- Project: Representation Alignment for MLIP Uncertainty
- Institution: SNU CCEL
- Purpose: Academic research on uncertainty quantification in materials science

---

*Note: This issue blocks latent vector extraction for SevenNet family models in our MLIP comparison study.*