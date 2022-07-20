"""
A sample main function to run this program. You can also use the Runner
in your own context or pass it dicts to dynamically create configs.

Theo Rieken
"""
from src.Runner.Runner import Runner
import sys

# Main guard for multithreading the runner "below"
if __name__ == "__main__":

    # Extract the arguments
    jobs = sys.argv[1:]

    # Create a runner instance and pass it the jobs
    worker = Runner(jobs=jobs)

    # Start working on the jobs until all are finished
    worker.run()
