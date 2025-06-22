"""
Validation script that compares kinematics predictions against Isaac Sim ground truth.
"""
import numpy as np
import jax.numpy as jnp
from kinematics import fk, inverse_kinematics
from omni.isaac.core import World

def validate_fk_ik(n_samples=100):
    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(urdf_path="path/to/leg.urdf", translation=(0,0,0))
    fk_errors = []
    ik_errors = []
    for _ in range(n_samples):
        θ_rand = np.random.uniform(low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0])
        world.step()
        robot.set_joint_positions({'hip': θ_rand[0], 'knee': θ_rand[1], 'ankle': θ_rand[2]})
        world.step()
        foot_pose = robot.get_rigid_body_pose('foot_link')
        sim_pos = jnp.array(foot_pose.translation)

        # Forward Kinematics error
        pred_pos = fk(jnp.array(θ_rand))
        fk_error = jnp.linalg.norm(pred_pos - sim_pos)
        fk_errors.append(float(fk_error))

        # Inverse Kinematics error
        θ_init = jnp.zeros(3)
        θ_sol = inverse_kinematics(sim_pos, θ_init)
        robot.set_joint_positions({'hip': float(θ_sol[0]), 'knee': float(θ_sol[1]), 'ankle': float(θ_sol[2])})
        world.step()
        foot_pose_ik = robot.get_rigid_body_pose('foot_link')
        sim_pos_ik = jnp.array(foot_pose_ik.translation)
        ik_error = jnp.linalg.norm(sim_pos_ik - sim_pos)
        ik_errors.append(float(ik_error))

    print(f"FK mean error: {np.mean(fk_errors):.6f} m, max error: {np.max(fk_errors):.6f} m")
    print(f"IK mean error: {np.mean(ik_errors):.6f} m, max error: {np.max(ik_errors):.6f} m")

if __name__ == "__main__":
    validate_fk_ik()
