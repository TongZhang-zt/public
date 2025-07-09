from pathlib import Path
import numpy as np
import pinocchio as pin

# Load the model from a URDF file
pinocchio_model_dir = Path(__file__).parent / "models/"
model_path = pinocchio_model_dir / "dofbot_info"
mesh_dir = pinocchio_model_dir
urdf_filename = "dofbot.urdf"
urdf_model_path = model_path / "urdf/" / urdf_filename
model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=mesh_dir)

# Build a data frame associated with the model
data = model.createData()

# Print all available frames in the model
print("Available frames in the model:")
for i in range(model.nframes):
    print(f"{i}: {model.frames[i].name}")

# Define the end-effector (tool) frame ID
# Note: You need to replace TOOL_FRAME_ID with the actual frame ID from your model
# You can get the frame ID using: frame_id = model.getFrameId("your_frame_name")
TOOL_FRAME_ID = model.getFrameId("end_effector")  # Change to your end-effector frame name

# target_position = np.array([-0.08, 0.105, 0.04])  # Modify as needed
def inverse(position):
    # Set target position in the world frame (x, y, z)
    target_position = np.array(position)  # Modify as needed

    # Initial joint configuration
    q_init = pin.neutral(model)  # Start with neutral (zero) configuration

    # Setup IK problem
    q = q_init.copy()
    eps = 1e-4  # Precision
    IT_MAX = 3000  # Max number of iterations
    DT = 1e-1  # Step size
    damp = 1e-12  # Damping coefficient

    # Iterative inverse kinematics algorithm
    success = False
    for i in range(IT_MAX):
        # Forward kinematics: compute current position
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # Get current end-effector position
        current_position = data.oMf[TOOL_FRAME_ID].translation

        # Compute error
        error = target_position - current_position
        error_norm = np.linalg.norm(error)

        if error_norm < eps:
            success = True
            break

        # Compute Jacobian at current configuration (position only)
        J = pin.computeFrameJacobian(model, data, q, TOOL_FRAME_ID, pin.LOCAL_WORLD_ALIGNED)[:3, :]

        # Compute velocity using pseudo-inverse of Jacobian
        v = np.linalg.solve(J.dot(J.T) + damp * np.eye(3), J.dot(J.T).dot(error))
        dq = J.T.dot(v) * DT

        # Update configuration
        q = pin.integrate(model, q, dq)

        if i % 10 == 0:
            print(f"Iteration {i}: error = {error_norm:.6f}")

    # Print results
    if success:
        print("\nInverse kinematics converged!")
        print(f"Final error: {error_norm:.6f}")
    else:
        print("\nInverse kinematics did not converge within max iterations.")
        print(f"Final error: {error_norm:.6f}")

    print("\nTarget position:", target_position)
    print("Final position:", data.oMf[TOOL_FRAME_ID].translation)
    print("Joint angles (in radians):", q)
    return q

