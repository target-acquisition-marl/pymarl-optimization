#!/bin/bash

MAX_JOBS=${1:-2}
YAML_FILE="queue.yaml"

conda init
conda activate pymarl-venv
mkdir -p logs

# parse YAML configs
COMMANDS=$(python3 <<EOF
import yaml

with open("$YAML_FILE") as f:
    data = yaml.safe_load(f)

for i, exp in enumerate(data.get('experiments', [])):
    try:
        config = exp['config']
    except KeyError:
        print(f"# Skipping experiment {i}: missing 'config'", flush=True)
        continue

    env_args = exp.get('env_args', {})
    other_args = {k: v for k, v in exp.items() if k not in ['config', 'env_args']}

    def to_cli_args(d, prefix=""):
        args = []
        for k, v in d.items():
            # val = str(v).lower() if isinstance(v, bool) else v
            val = v
            args.append(f"{prefix}{k}={val}")
        return args

    all_args = to_cli_args(other_args) + to_cli_args(env_args, prefix="env_args.")
    cmd = f"python3 src/main.py --config={config} --env-config=sc2 with {' '.join(all_args)} \n"
    print(cmd)
EOF
)


# Loop through commands, queue jobs
while IFS= read -r CMD; do
  [[ -z "$CMD" ]] && continue

  while [[ $(jobs -rp | wc -l) -ge $MAX_JOBS ]]; do
    sleep 2
  done

  echo "Starting: $CMD"
  log_file="logs/$(date +%F_%H-%M-%S).log"
  nohup bash -c "$CMD" > "$log_file" 2>&1 &
done <<< "$COMMANDS"

wait
echo "All experiments completed."
