# Reinforcement Learning with LQR

Golf Team project, created for AI For Robotics course, implements a Reinforcement Learning (RL) training pipeline using Linear Quadratic Regulator (LQR) techniques. The `main.py` script serves as the entry point to start the training process, leveraging the required libraries specified in `requirements.txt`.

## Authors

- Ida Bagus Kade Rainata Putra - M11312806
- Ingwer Ludwig Nommensen - M11302839
- 郭志豪 (Chih-Hao, Kuo) - M11312006

## Prerequisites

Ensure you have the following installed:
- Python 3.10 or higher
- `pip` (Python package manager)

## Setup Instructions

Follow these steps to set up and run the training:

1. **Clone the Repository** (if applicable):
   If this project is hosted in a repository, clone it to your local machine:
   ```bash
   git clone https://github.com/ingwerludwig/td3-rl-quadcopter.git
   ```

2. **Install Dependencies**:
   Install the required Python libraries listed in `requirements.txt` using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   This command installs all necessary packages, ensuring the environment is ready for training.


3. **Install Simulation Gym Pybullet Dependencies**:
   Start the RL training with LQR by executing script:
   ```bash
   cd simulation/gym-pybullet-drones
   ```
   ```bash
   pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`
   ```
   This command initiates the training process, which will handle all necessary simulation process.
   Please refer to this repository when installing the simulation.
   ```bash
   https://github.com/utiasDSL/gym-pybullet-drones?tab=readme-ov-file
   ```

4. **Run the Training**:
   Start the RL training with LQR by executing script:
   ```bash
   bash ./scripts/run_training.sh
   ```
   This command initiates the training process, which will handle all necessary computations, agent training, and LQR optimization.


5. **Run the Simulation**:
   Start the Quadcopter Simulation of Gym Pybullet by executing script:
   ```bash
   bash ./scripts/run_simulation.sh
   ```
   This command initiates all necessary computations environment setup.


## Notes
- Ensure you are in the project directory when running the commands.
- If you encounter issues with missing dependencies, verify that `requirements.txt` is complete and matches your environment's Python version.
- For detailed logs or outputs, check the console or any generated files as specified in `main.py`.

## Troubleshooting
- **ModuleNotFoundError**: Ensure all dependencies are installed correctly by re-running `pip install -r requirements.txt`.
- **Python Version Issues**: Confirm your Python version with `python --version` and consider using a virtual environment to avoid conflicts.