from joblib import Parallel, delayed
import subprocess
import itertools
# Function to run a command
def run_command(arg1, arg2):
    result = subprocess.run(["python", "macrocircuits_parallel.py", arg1, arg2], capture_output=True, text=True)
    return result.stdout

if __name__ == "__main__":
    # List of argument pairs to pass to the script

    i=list(range(10))
    j=[1]
    i=[str(x) for x in i]
    j=[str(x) for x in j]
    combs=itertools.product(i,j)

    # Use joblib to run commands in parallel
    results = Parallel(n_jobs=10)(delayed(run_command)(arg1, arg2) for arg1, arg2 in combs)

    # Print the results
    for output in results:
        print(output)
