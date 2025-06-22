<<<<<<< HEAD
# isaac_leg_project
=======
# 3-DOF Robot Leg Kinematics

This project implements forward and inverse kinematics for a planar 3-DOF robot leg using SymPy and JAX, and validates the results against NVIDIA Isaac Sim ground truth.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Forward/Inverse Kinematics Module**  
   - `src/kinematics.py` provides:
     - `fk(thetas: jnp.ndarray[3]) -> jnp.ndarray[3]`
     - `jacobian(thetas: jnp.ndarray[3]) -> jnp.ndarray[3,3]`
     - `inverse_kinematics(target: jnp.ndarray[3], θ_init: jnp.ndarray[3]) -> jnp.ndarray[3]`
     - `batch_fk` and `batch_ik` for vectorized operations.

2. **Validation in Isaac Sim**  
   - `src/isaac_validation.py` runs `n_samples` of random tests:
     - Compares FK predictions to simulated foot positions.
     - Measures IK convergence and end-effector error.
   - Adjust the URDF path in the script before running.

3. **Launch Script**  
   - Use `launch.sh` to activate the environment and run the validation script in headless Isaac Sim.

## Files

- `src/kinematics.py`: Kinematics implementation.
- `src/isaac_validation.py`: Validation against Isaac Sim.
- `requirements.txt`: Python dependencies.
- `launch.sh`: Helper to start Isaac Sim validation.
- `README.md`: This documentation.
>>>>>>> d125b8f (first commit)
