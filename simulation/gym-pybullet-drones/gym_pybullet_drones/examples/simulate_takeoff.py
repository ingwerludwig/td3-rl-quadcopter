import time
import argparse
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.control.LQRControl import *
from src.agents.td3 import TD3
from pprint import pprint

# ---------- DEFAULTS ----------
DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB
    ):

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0, 0, 0]])  # Start at origin
    env = CtrlAviary(
        drone_model=drone,
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB,  # Basic physics
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=False
    )

    #### Initiate: Quadcopter Config and ENV #####################
    quad_config = Quadcopter()
    lqr_env = QuadcopterLQREnv(quadcopter=quad_config)

    #### Load: checkpoint model#####################
    state_dim = lqr_env.observation_space.shape[0]
    action_dim = lqr_env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3(state_dim, action_dim, device)
    agent.load_checkpoint("/home/citiai-cygnus/VisionRAG-Ingwer/TugasIngwer/TugasAIForRobotics/src/FIXED RESULT/HOVER/checkpoint/td3_quadcopter_epoch_100.pth")

    #### Trajectory: smooth vertical climb #####################
    NUM_WP = control_freq_hz * duration_sec
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i] = [0, 0, 3]  # Linear climb from 0 to 2 meters

    wp_counter = 0


    #### Controller + Logger ###################################
    ctrl = [DSLPIDControl(drone_model=drone)]
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    duration_sec=duration_sec,
                    output_folder=output_folder,
                    colab=colab)

    #### Run the simulation loop ###############################
    action = np.zeros((1, 4))
    obs = env.reset()
    start_time = time.time()

    ####
    action_store = []

    for i in range(0, int(duration_sec * control_freq_hz)):

        #### Step the simulation ################################

        # obs, reward, terminated, truncated, info = lqr_env.step()
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute LQR control ###############################
        # print(f"This {TARGET_POS}")
        st = obs[0]
        # x, x rates, y, y rates, z, z rates, x dot, x dot rates, y dot, y dot rates, z dot, z dot rates
        reformatted_curr_state = np.array(
            [st[0], st[10], st[1], st[11], st[2], st[12], st[3], st[13], st[4], st[14], st[5], st[15]])
        reformatted_target_state = np.array([TARGET_POS[wp_counter][0], 0, TARGET_POS[wp_counter][1], 0, TARGET_POS[wp_counter][2], 0, 0, 0, 0, 0, 0, 0])

        lqr_env.states = reformatted_curr_state
        lqr_env.set_ref_states(reformatted_target_state)


        lqr_params = agent.select_action(reformatted_curr_state - reformatted_target_state)
        lqr_env.update_lqr_params(lqr_params)

        actions = lqr_env.compute_quad_actions_from_controller()
        action = np.array([lqr_env._convert_to_rpm(actions)])

        # Simpen action --> append action store
        # Compute lqr params: agent select action from reformatted_curr_state vs reformatted_target_state
        # Update lqr params : lqr_env.update_lqr_params

        #### Advance waypoint ###################################
        wp_counter = wp_counter + 1 if wp_counter < (NUM_WP - 1) else wp_counter

        #### Logging ############################################
        logger.log(
            drone=0,
            timestamp=i / control_freq_hz,
            state=obs[0],
            control=np.hstack([TARGET_POS[wp_counter], np.zeros(9)])
        )

        #### Render + Real-time sync ############################
        env.render()
        if gui:
            sync(i, start_time, 1 / control_freq_hz)

    #### Cleanup ###############################################
    env.close()
    logger.save()
    logger.save_as_csv("takeoff_z")
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vertical takeoff simulation")
    parser.add_argument('--drone',              default=DEFAULT_DRONE, type=DroneModel, help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI, type=str2bool, help='Use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC, type=int, help='Duration in seconds (default: 12)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder for log results (default: results)', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=str2bool, help='If running in Google Colab (default: False)', metavar='')
    args = parser.parse_args()

    run(**vars(args))