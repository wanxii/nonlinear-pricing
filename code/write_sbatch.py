import sys
import os

def main(rate_c1, end):
    s = '''
        #!/bin/bash
        #SBATCH --job-name=c1_{}_end_{}       # job name
        #SBATCH --output=c1_{}_end_{}.out     # output log file
        #SBATCH --error=c1_{}_end_{}.err      # error file
        #SBATCH --time=8:00:00
        #SBATCH --nodes=1           # 1 GPU node
        #SBATCH --ntasks=1          # 1 CPU core to drive GPU
        #SBATCH --cpus-per-task=4
        #SBATCH --mem-per-cpu=8G

        module load miniconda
        conda activate jupyterlab

        python ./c1_end.py {} {} >> c1_{}_end_{}.out
        '''.format(rate_c1, end, rate_c1, end, rate_c1, end,
                   rate_c1, end, rate_c1, end, rate_c1, end)
    with open("c1_{}_end_{}.sbatch".format(rate_c1, end), "w") as f:
        for l in s.strip().split("\n"):
            f.write(l.strip() + "\n")
    print("Created file", "c1_{}_end_{}.sbatch".format(rate_c1, end))

if __name__ == "__main__":
    rate_c1 = sys.argv[1]
    end = sys.argv[2]
    main(rate_c1, end)