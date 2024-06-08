import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--number", type=str)
args = parser.parse_args()
print(f'{args.number} got')
