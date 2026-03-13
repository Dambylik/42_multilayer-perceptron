import random
import csv
import sys


def list_shuffle(items):
    random.seed(42)
    for i in range(len(items) - 1, 0, -1):
        pick = random.randint(0, i)
        items[pick], items[i] = items[i], items[pick] 
    return items

print(list_shuffle(["A", "B", "C", "D"]))

def load_csv(filename):
    X = []
    y = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)

            for line_num, row_data in enumerate(reader, 1):
                try:
                    label = 1 if row_data[1] == 'M' else 0
                    features = [float(val) for val in row_data[2:]]
                    X.append(features)
                    y.append(label)
                except ValueError:
                    print(f"Can not convert data to numbers in line {line_num}")

    except FileNotFoundError:
        print(f"File '{filename}' was not found")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occured: {e}")
        sys.exit(1)    
    return X, y


def main():
    data_x, data_y = load_csv('data.csv')
    print(data_x)
    print(len(data_y))


if __name__ == "__main__":
    main()        