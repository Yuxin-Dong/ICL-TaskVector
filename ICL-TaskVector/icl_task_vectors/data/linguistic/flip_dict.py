import json
import sys

def flip_dict(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # Flip the key-value pairs
    flipped = {value: key for key, value in data.items()}

    # Write the flipped dictionary to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(flipped, outfile, indent=4)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python flip_dict.py <input_file> <output_file>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    flip_dict(input_filename, output_filename)
