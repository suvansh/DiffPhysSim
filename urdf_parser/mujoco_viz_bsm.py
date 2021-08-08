import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer


bsm_path = '/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/xmls/bsm_new.xml'
tosser_path = '/Users/sanjeev/GoogleDrive/CMU/Research/DiffPhysSim/urdf_parser/xmls/tosser.xml'

def sim(path, timesteps, ctrl=None, view=False):
    model = load_model_from_path(path)
    sim = MjSim(model)
    qs = [sim.data.qpos]
    if ctrl is None:
        ctrl = np.zeros(timesteps)
    elif type(ctrl) is int:
        ctrl *= np.ones(timesteps)
    if view:
        viewer = MjViewer(sim)
    # simulation loop
    for t in range(timesteps):
        sim.data.ctrl[:] = ctrl[t]
        sim.step()
        qs.append(sim.data.qpos)
        if view:
            viewer.render()
    return qs

timesteps = 5000
ctrl_block = -5*np.ones((250, 4))
ctrl = np.concatenate((ctrl_block, -ctrl_block, ctrl_block, -ctrl_block, np.zeros((4000, 4))))
qs = sim(bsm_path, timesteps, ctrl=ctrl, view=True)
breakpoint()
