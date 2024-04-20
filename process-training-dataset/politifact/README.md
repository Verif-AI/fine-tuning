### Commands

fine-tuning/process-training-dataset/politifact

$ python process_data_for_fine_tuning.py > processed-data/politifact_for_fine_tuning.jsonl

fine-tuning/process-training-dataset/politifact

$ sh split_data_for_training_and_testing.sh

### Upload to AWS S3

https://docs.aws.amazon.com/AmazonS3/latest/userguide/upload-objects.html


### Fine tuning job

https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/custom-models
