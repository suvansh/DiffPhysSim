using MeshCat
using RigidBodyDynamics
using MeshCatMechanisms

vis = Visualizer()
render(vis)

bsm_path = "/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/bsm.urdf"
robot = parse_urdf(bsm_path)
delete!(vis)
mvis = MechanismVisualizer(robot, URDFVisuals(bsm_path), vis)
