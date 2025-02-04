import traceback
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

found_dotenv = find_dotenv()
if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

# get folder of found_dotenv
env_path = Path(found_dotenv).parent
sys.path.insert(0, str(env_path))
print(f"env_path: {env_path}")
print(f"sys.path: {sys.path}")

try:
    from app_modules.utils import *
    from eval_modules.calc_repetitions_v2 import *
except ModuleNotFoundError as e:
    print("ModuleNotFoundError:", e)
    traceback.print_exc()

load_ms_marco_result(ms_marco_csv_result_files, calc_bertscore=True)
