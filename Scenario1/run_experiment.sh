#!/bin/bash
 
# Activate your environment if needed
# source ~/myenv/bin/activate
 
# Path to your Python script
SCRIPT_NAME="main.py"
 
# Common arguments
PERSONAS_FILE="personas.json"
MODE="2vs2"
MAX_ROUND=10
EXP_NUM=10
START_EXP=0
 
# Team definitions
T1="Rachel Stein,Paola Wei"
T1_COPY="Rachel Stein Copy,Paola Wei Copy"
 
T2="Roxanne Gray,Selina Rhodes"
T2_COPY="Roxanne Gray Copy,Selina Rhodes Copy"
 
T3="Graham Wood,Robbie Cruz"
T3_COPY="Graham Wood Copy,Robbie Cruz Copy"
 
# Function to run experiment
run_experiment() {
  local team1_name=$1
  local team2_name=$2
  local team1=$3
  local team2=$4
  local out_dir=$5
 
  echo "Running $team1_name vs $team2_name"
  python $SCRIPT_NAME \
    --mode $MODE \
    --team1_personas "$team1" \
    --team2_personas "$team2" \
    --personas_file $PERSONAS_FILE \
    --output_dir "$out_dir" \
    --max_round $MAX_ROUND \
    --start_exp $START_EXP \
    --exp_num $EXP_NUM
}
 


# Run t1 vs t2
run_experiment "t1" "t2" "$T1" "$T2" "t1vst2"

# Run t1 vs t3
run_experiment "t1" "t3" "$T1" "$T3" "t1vst3"

# Run t2 vs t3
run_experiment "t2" "t3" "$T2" "$T3" "t2vst3"

# Run t3 vs t3 with T3_COPY
run_experiment "t3" "t3" "$T3" "$T3_COPY" "t3vst3"

# Run t1 vs t1 with T1_COPY
run_experiment "t1" "t1" "$T1" "$T1_COPY" "t1vst1"

# Run t2 vs t2 with T2_COPY
run_experiment "t2" "t2" "$T2" "$T2_COPY" "t2vst2"

 
echo "All experiments completed!"