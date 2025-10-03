import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as cs
import os
import json

# === Load model from updated URDF ===
# Get project root (1 level up from notebook directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Build path to URDF inside src/
urdf_path = os.path.join(project_root,"pendulum_model", 
                         "NEW_PENDULUM_DESCRIPTION", "NEW_furuta_pendulum_Q4.urdf")

print("URDF path:", urdf_path)
urdf_path = os.path.abspath(urdf_path)

# Build standard model
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Build CasADi symbolic model
cmodel = cpin.Model(model)
cdata = cmodel.createData()

# === Print DoF check ===
print(f"Model name: {model.name}")
print(f"Degrees of freedom: nq = {model.nq}, nv = {model.nv}")

# === Define symbolic variables ===
q = cs.SX.sym("q", model.nq)
v = cs.SX.sym("v", model.nv)
tau_in = cs.SX.sym("tau_in", 1)  # symbolic input: 1D torque

# === Build full tau vector ===
tau = cs.vertcat(tau_in, cs.SX.zeros(model.nv - 1))

# === Dynamics computation ===
M = cpin.crba(cmodel, cdata, q)
C = cpin.computeCoriolisMatrix(cmodel, cdata, q, v)
g = cpin.computeGeneralizedGravity(cmodel, cdata, q)

qdd = cs.solve(M, tau - C @ v - g)

# === Define function ===
f_dyn = cs.Function("forward_dynamics", [q, v, tau_in], [qdd])

# Save
with open("NEW_forward_dynamics_Q4.casadi", "w") as f:
    f.write(f_dyn.serialize())

print("Saved as NEW_forward_dynamics_Q4.casadi")
