"""
Kinematics module for a 3-DOF planar robot leg using Sympy and JAX.
"""

import sympy as sp
import jax.numpy as jnp
from jax import jit, lax
from jax import jacfwd
from jax import vmap

# Link lengths (meters)
L1 = 0.1
L2 = 0.2
L3 = 0.05

# Symbolic variables
θ1, θ2, θ3 = sp.symbols('θ1 θ2 θ3')
l1, l2, l3 = sp.symbols('l1 l2 l3', positive=True)

#  angles
a1 = θ1
a2 = θ1 + θ2
a3 = θ1 + θ2 + θ3

# Symbolic forward kinematics (planar, z=0)
x = l1*sp.cos(a1) + l2*sp.cos(a2) + l3*sp.cos(a3)
y = l1*sp.sin(a1) + l2*sp.sin(a2) + l3*sp.sin(a3)
pos = sp.Matrix([x, y, 0])

# Symbolic Jacobian
J_sym = pos.jacobian([θ1, θ2, θ3])

# Lambdify to JAX functions
fk_fn = sp.lambdify((θ1, θ2, θ3, l1, l2, l3), pos, 'jax')
J_fn  = sp.lambdify((θ1, θ2, θ3, l1, l2, l3), J_sym, 'jax')

@jit
def fk(thetas):
    """
    Forward kinematics: returns [x, y, 0] for given thetas [θ1, θ2, θ3].
    """
    θ1_, θ2_, θ3_ = thetas
    p = fk_fn(θ1_, θ2_, θ3_, L1, L2, L3)
    return jnp.array(p)

@jit
def jacobian(thetas):
    """
    Analytic Jacobian matrix (3x3) for given thetas.
    """
    θ1_, θ2_, θ3_ = thetas
    Jmat = J_fn(θ1_, θ2_, θ3_, L1, L2, L3)
    return jnp.array(Jmat)

# Damped least squares parameters
_damping = 1e-2

def _ik_iteration(carry, _):
    θ, target = carry
    p = fk(θ)
    err = target - p
    Jmat = jacobian(θ)
    JTJ = Jmat.T @ Jmat + (_damping**2)*jnp.eye(3)
    delta = jnp.linalg.solve(JTJ, Jmat.T @ err)
    return (θ + delta, target), None

def inverse_kinematics(target, θ_init, n_iters=20):
    """
    Inverse kinematics via damped least squares.
    Returns solution thetas for desired 3D target position.
    """
    (θ_sol, _), _ = lax.scan(_ik_iteration, (θ_init, target), None, length=n_iters)
    return θ_sol

# Batch (vectorized) versions
batch_fk = vmap(fk, in_axes=(0,))
batch_ik = vmap(inverse_kinematics, in_axes=(0, 0))
