#!/bin/bash

MAX_JOBS=${1:-2}
YAML_FILE="queue.yaml"

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
        map_name = exp['map_name']
        mask_prob = exp['mask_prob']
        is_sticky = exp['is_sticky']
    except KeyError as e:
        print(f"# Skipping experiment $i: missing key {e}", flush=True)
        continue

    cmd = f"python3 src/main.py --config={config} --env-config=sc2 with env_args.map_name={map_name} mask_prob={mask_prob} is_sticky={is_sticky} \n"
    # cmd = "echo 44 && sleep 6 \n"
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

