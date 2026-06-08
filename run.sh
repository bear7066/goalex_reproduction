if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET=$1


{
  python src/iterative_cluster.py \
    --data_path processed_data/$DATASET \
    --exp_dir experiments/$DATASET \
    --proposer_model oss120b \
    --assigner_name oss20b \

    --proposer_num_descriptions_to_propose 30 \
    --assigner_for_final_assignment_template templates/t5_multi_assigner_one_output.txt \
    --iterative_max_rounds 2 \
    --turn_off_approval_before_running \
    --verbose
    
  python src/evaluate_only.py \
    --data_path processed_data/$DATASET \
    --exp_dir experiments/$DATASET
} 2>&1 | tee "experiments/$DATASET/log_$(date +%Y%m%d_%H%M%S).txt"
