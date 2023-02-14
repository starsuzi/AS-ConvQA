import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--original_train_path", default="./datasets/quac/train.json", type=str, help="path")
parser.add_argument("--nbest_pred_1_path", default="", type=str, help="path")
parser.add_argument("--output_step2_train_path", default="", type=str, help="path")
args = parser.parse_args()

with open(args.original_train_path) as json_file, open(args.nbest_pred_1_path) as predicted_json_file:
    json_data = json.load(json_file)
    json_predicted_data = json.load(predicted_json_file)
    
    for i, data in enumerate(json_data['data']):
        for paragraph in data['paragraphs']:
            for qa in paragraph['qas']:
                qa['predicted_answers']=json_predicted_data[qa['id']]

with open(args.output_step2_train_path, "w") as writer: 
    writer.write(json.dumps(json_data, indent=4) + "\n")

