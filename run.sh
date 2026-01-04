if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET=$1


{
  python src/iterative_cluster.py \
    --data_path processed_data/$DATASET \
    --exp_dir experiments/$DATASET \
    --proposer_model gpt-oss-20b \
    --assigner_name gpt-oss-20b \
    --proposer_num_descriptions_to_propose 30 \
    --assigner_for_final_assignment_template templates/t5_multi_assigner_one_output.txt \
    --iterative_max_rounds 2 \
    --verbose
    
  LOG_FILE="experiments/$DATASET/log_$(date +%Y%m%d_%H%M%S).txt"

  python src/evaluate_only.py \
    --data_path processed_data/$DATASET \
    --exp_dir experiments/$DATASET
} 2>&1 | tee "$LOG_FILE"
