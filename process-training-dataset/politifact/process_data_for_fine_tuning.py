import json
import sys
sys.stdout.reconfigure(encoding='utf-8')


def process_data_for_fine_tuning(filename):
    with open(filename, encoding="utf-8") as file:
        json_file = json.load(file)

        for idx, json_doc in enumerate(json_file):
            claim = json_doc["claim"]
            veracity = json_doc["veracity"]

            if veracity in ["pants-fire", "false", "true"]:
                # build training data
                prompt = "Please do fact checking for " + claim
                if veracity in ["pants-fire", "false"]:
                    completion = "objective false"
                elif veracity == "true":
                    completion = "objective truth"
                training_doc = {"prompt": prompt, "completion": completion}
            
                print(json.dumps(training_doc))

# process training data
SOURCE_FILE = "data/politifact_3000.json"
process_data_for_fine_tuning(SOURCE_FILE)
