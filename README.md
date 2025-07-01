```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in SMAC (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232 not SC2.4.10.
```

# Python MARL framework
This is a fork of PyMARL with modifications to support agent-level masking in the mixer network (in QMIX for now).

PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Run an experiment 

```shell
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

# Modifications

## Random masking QMix:

Introduced masking mechanisms for agent Q-values before mixing:
  - random masking: agents' Q-values are randomly masked at each timestep with a given probability.

```shell
python3 src/main.py --config=qmix_masked --env-config=sc2 with mask_prob=0.5 masked_local_obs=False is_sticky=False is_fixed=False env_args.map_name=3s5z env_args.obs_instead_of_state=True 
```

## Experiment Queue with YAML Configuration

This project includes a simple experiment queue system that runs multiple ML experiments **with controlled concurrency**.

### How It Works

- **Experiment configurations** are stored in a `queue.yaml` file in a clean, human-readable format.
- The runner script `run_queue.sh` reads this YAML file, converts each experiment config to a full command, and runs up to `MAX_JOBS` experiments in parallel.
- Once a running experiment finishes, the next one in the queue starts automatically.
- All experiment outputs are logged into the `logs/` directory, each with a timestamped log file.
- The script uses `nohup` so it continues running even if the terminal session is closed.

### Example `queue.yaml`

```yaml
experiments:
  - config: qmix_masked
    map_name: 1c3s5z
    mask_prob: 0.6
    is_sticky: true

  - config: qmix_masked
    map_name: 2s3z
    mask_prob: 0.4
    is_sticky: false
```

### Running Experiments

Run the queue with:

```bash
chmod +x run_queue.sh
nohup ./run_queue.sh $MAX_JOBS &
```

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Documentation/Support

Documentation is a little sparse at the moment (but will improve!). Please raise an issue in this repo, or email [Tabish](mailto:tabish.rashid@cs.ox.ac.uk)

## Citing PyMARL 

If you use PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

## License

Code licensed under the Apache License v2.0
