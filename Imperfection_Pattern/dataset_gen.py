import argparse
from data.preprocess import SETGENERATOR
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Dataset Generator")

parser.add_argument("--dataset", required=True, type=str,
                    choices=["BPIC11.csv", "BPIC15_1.csv", "credit-card-new.csv", "pub-new.csv"],help="event log name")

parser.add_argument("--filepath", type=str, 
    default='data/datasets/',  help="dataset location")

parser.add_argument("--task", required=True, type=str,
                    choices=["next_activity", "outcome", "event_remaining_time", "case_remaining_time"],
                    help="task name (one of: next_activity, outcome, event_remaining_time, case_remaining_time)")

parser.add_argument("--saveloc", type=str, 
    default='data/processed/',  help="preprossed dataset save location")


args = parser.parse_args()

if __name__ == "__main__": 
    set_generator = SETGENERATOR(args.dataset, args.filepath, args.task, args.saveloc)
    set_generator.SetGenerator()