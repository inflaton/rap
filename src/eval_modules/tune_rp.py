import datetime
import os
import subprocess
import sys
from timeit import default_timer as timer

sys.path.insert(0, os.getcwd())

start = timer()

result_filename_prefix = os.getenv("RESULT_FILENAME_PREFIX")
if not result_filename_prefix:
    now = datetime.datetime.now()
    result_filename_prefix = "Tune_{:%Y-%m-%d_%H-%M-%S}".format(now)

print(f"Result filename prefix: {result_filename_prefix}")
all_results_filename = f"./data/results/{result_filename_prefix}.csv"

repetition_penalty_delta = 0.02
repetition_penalty_end = 1.3
repetition_penalty = 1.0

repetition_penalty_start = os.getenv("REPETITION_PENALTY_START")
if repetition_penalty_start:
    repetition_penalty = float(repetition_penalty_start)
    print(f"Starting from RP: {repetition_penalty}")

while repetition_penalty <= repetition_penalty_end + 1e-5:
    new_env = os.environ.copy()

    repetition_penalty_str = f"{repetition_penalty:.3f}"
    new_env["HFTGI_RP"] = repetition_penalty_str
    new_env["HF_RP"] = repetition_penalty_str
    new_env["ML_RP"] = repetition_penalty_str
    new_env["SL_RP"] = repetition_penalty_str

    log_file = "./data/logs/{}_RP_{}.txt".format(
        result_filename_prefix, repetition_penalty_str
    )
    test_results_filename = "./data/results/{}_RP_{}.csv".format(
        result_filename_prefix, repetition_penalty_str
    )
    new_env["TEST_RESULTS_CSV_FILE"] = test_results_filename
    new_env["ALL_RESULTS_CSV_FILE"] = all_results_filename

    num_questions = os.getenv("NUM_QUESTIONS") or ""

    with open(log_file, "w") as f_obj:
        subprocess.run(
            f"python qa_chain_test.py {num_questions}",
            shell=True,
            env=new_env,
            stdout=f_obj,
            text=True,
        )

    repetition_penalty += repetition_penalty_delta

print(f"All results saved to {all_results_filename}")

end = timer()
total_time = end - start
print(f"Total time used: {total_time:.3f} s")
