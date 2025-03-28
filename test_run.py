import csv
import subprocess
import sys

def main(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                cmd = [
                    'python', 'run.py',
                    '--model', 'SepReformer_Base_WSJ0',
                    '--engine-mode', 'infer_sample',
                    '--sample-file', row['mixture'],
                    '--ref', row['enroll'],
                    '--src-sample', row['source']
                ]
                print("Executing:", ' '.join(cmd))
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed for row {reader.line_num}: {e}")
            except KeyError as e:
                print(f"Missing column in CSV: {e}")
                sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_csv>")
        sys.exit(1)
    main(sys.argv[1])