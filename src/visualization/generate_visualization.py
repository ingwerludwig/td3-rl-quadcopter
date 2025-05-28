import numpy as np

from src.visualization.visualization_utils import *
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from io import BytesIO

def generate_quad_sim(state, reference_state, episode, save_filename):
    plot_results(state, reference_state, episode, save_filename)

def generate_video_plot(states, ref_states, save_file_name):
    frames = []
    start_point = np.array([
        states[0][0], states[0][2], states[0][4]
    ])
    end_point = np.array([
        states[-1][0], states[-1][2], states[-1][4]
    ])
    ref_states = np.array(ref_states)
    sequence = []
    for i,st in enumerate(states):
        print(f"Generating: {i+1}/{len(states)} frames")
        sequence.append(st)
        fig = plot_singluar_vid(
            drone_history=np.array(sequence),
            ref_states=ref_states,
            start_point=start_point,
            end_point=end_point,
            trajectory_name="Line"
        )
        buf = BytesIO()
        fig.savefig(buf, dpi=300, bbox_inches='tight', format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    imageio.mimsave(save_file_name, frames, fps=60)



