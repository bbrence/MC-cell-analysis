import pickle
import sys

def print_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
        
    print(data)

if __name__ == "__main__":
    print_pkl(sys.argv[1])