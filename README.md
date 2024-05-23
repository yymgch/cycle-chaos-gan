This repository is the official implementation of the paper "Cyclic image generation using chaotic dynamics" by Takaya Tanaka and Yutaka Yamaguti.

This repository contains codes to define and train CycleChaosGAN models for generating 
sequence of images, to analyze the generated sequences from the viewpoint of chaos theory, and to visualize the results.



## files in this directory

- `training.py`: contains the main training loop for our model.
- `model.py`: contains definition of the model
- `util.py`: contains utility functions
- `parameters.py`: contains parameters setting
- `lyapunov.py`: contains the code for calculating Lyapunov Exponents
- `umap_train_single.py`: contains the code for visualization of single trajectory using a UMAP model
- `umap_train_many.py`: contains the code for visualization of many trajectories using a UMAP model
- `lyapunov_exe.py`: made a figure for Lyapunov analysis
- `precision_recall.py`: contains the code for precision/recall analysis
- `makefig_trj.py`: contains the code for making the image sequence figure
- `makefig_lyapunov.py`: contains the code for making the Lyapunov exponents figures
- `eval_pre_rec.py`: contains the code for calculating the precision/recall values of from single trajectory
- `eval_many_pre_rec.py`: contains the code for calculating the precision/recall values of from set of states
- `lyapunov_direct.py`: contains the code for calculating the largest Lyapunov Exponents directly evaluating the diverge of the trajectories
- `model_graph.py`: contains the code for making the model graph for visualization

## how to reproduce figures in the paper

1. Install conda.
2. Move to the directory of this repository
3. Create an environment by by running `conda env create --file myenv.yaml`
4. Activate the environment by running `conda activate tf_chaos`
5. Run the shell './run_all.sh' to train the model and generate the figures 







