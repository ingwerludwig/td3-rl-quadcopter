import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os


def load_metrics(prefix, max_episode=100):
    """Load training or validation metrics from JSON files"""
    rewards = []
    critic_losses = []
    q_values = []
    actor_losses = []

    for ep in range(1, max_episode + 1):
        epoch_dir = os.path.join(
            "/home/citiai-cygnus/VisionRAG-Ingwer/TugasIngwer/TugasAIForRobotics/src/FIXED RESULT/LINE/results/metrics/episode")
        filename = f"Episode_{ep}_{prefix}_metrics.json"
        filename = os.path.join(epoch_dir, filename)
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                episode_key = f"Episode {ep}"
                rewards.append(data[episode_key]["Avg Reward"])
                critic_losses.append(data[episode_key]["Avg Critic Loss"])
                q_values.append(data[episode_key]["Avg Q"])
                actor_losses.append(data[episode_key]["Avg Actor Loss"])
        except FileNotFoundError:
            print(f"Warning: File {filename} not found")

    return {
        'rewards': np.array(rewards),
        'critic_losses': np.array(critic_losses),
        'q_values': np.array(q_values),
        'actor_losses': np.array(actor_losses),
        'episodes': np.arange(1, len(rewards) + 1)
    }


def main():
    # Load training and validation metrics
    training_metrics = load_metrics("training")
    validation_metrics = load_metrics("validation")

    # Create and save separate plots for each metric
    metrics = ['rewards', 'critic_losses', 'q_values', 'actor_losses']
    titles = ['Average Reward', 'Average Critic Loss', 'Average Q Value', 'Average Actor Loss']
    ylabels = ['Reward', 'Loss', 'Q Value', 'Loss']
    filenames = ['rewards_plot.png', 'critic_loss_plot.png', 'q_values_plot.png', 'actor_loss_plot.png']

    output_dir = "/home/citiai-cygnus/VisionRAG-Ingwer/TugasIngwer/TugasAIForRobotics/src/testing/"

    for metric, title, ylabel, filename in zip(metrics, titles, ylabels, filenames):
        plt.figure(figsize=(8, 5))
        plt.plot(training_metrics['episodes'], training_metrics[metric], 'b-', label='Training')
        plt.plot(validation_metrics['episodes'], validation_metrics[metric], 'r-', label='Validation')
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the current figure
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    main()