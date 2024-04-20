PROCESSED_DATA="processed-data/politifact_for_fine_tuning.jsonl"
TRAINING_DATA="processed-data/politifact_for_fine_tuning_training.jsonl"
TESTING_DATA="processed-data/politifact_for_fine_tuning_testing.jsonl"

head -n 50 $PROCESSED_DATA > $TESTING_DATA
sed '1,50d' $PROCESSED_DATA > $TRAINING_DATA