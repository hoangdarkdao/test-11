from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation
from llm4ad.task.optimization.bi_kp import BIKPEvaluation
from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation

from llm4ad.tools.llm.llm_api_codestral import MistralApi

from llm4ad.method.momcts import MOMCTS_AHD, MOMCTSProfiler
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler
from llm4ad.method.mpage import EoHProfiler, MPaGE
from llm4ad.method.moead import MOEAD, MOEADProfiler

# Import bounds from all supported tasks
from llm4ad.task.optimization.bi_tsp_semo.bound import UPPER_BOUND as TSP_UPPER, LOWER_BOUND as TSP_LOWER
from llm4ad.task.optimization.bi_kp.bound import UPPER_BOUND as KP_UPPER, LOWER_BOUND as KP_LOWER
# from llm4ad.task.optimization.bi_cvrp.bound import UPPER_BOUND as CVRP_UPPER, LOWER_BOUND as CVRP_LOWER

import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------
# Mapping configurations
# ------------------------
algorithm_map = {
    'momcts': (MOMCTS_AHD, MOMCTSProfiler),
    'meoh': (MEoH, MEoHProfiler),
    'nsga2': (NSGA2, NSGA2Profiler),
    'mpage': (MPaGE, EoHProfiler),
    'moead': (MOEAD, MOEADProfiler)
}

task_map = {
    "tsp_semo": BITSPEvaluation(),
    "bi_kp": BIKPEvaluation(),
    "bi_cvrp": BICVRPEvaluation(),
}

# Bounds per problem
bound_map = {
    "tsp_semo": (TSP_LOWER, TSP_UPPER),
    "bi_kp": (KP_LOWER, KP_UPPER)
    # "bi_cvrp": (CVRP_LOWER, CVRP_UPPER),
}

# ------------------------
# User configuration
# ------------------------
ALGORITHM_NAME = 'meoh'  # Could also be 'meoh', 'nsga2', 'mpage', 'moead'
PROBLEM_NAME = "bi_kp"  # Could also be 'bi_kp', 'bi_cvrp'
exact_log_dir_name = "v30"  # must be unique for each run
api_key = os.getenv('API_KEY3')  # change APIKEY1, APIKEY2, APIKEY3

# ------------------------
# Main execution
# ------------------------
if __name__ == '__main__':
    log_dir = f'logs/{ALGORITHM_NAME}/{PROBLEM_NAME}'

    MethodClass, ProfilerClass = algorithm_map[ALGORITHM_NAME]
    TaskClass = task_map[PROBLEM_NAME]
    LOWER_BOUND, UPPER_BOUND = bound_map[PROBLEM_NAME]  # ✅ Auto-select bounds

    llm = MistralApi(
        keys="",
        model='codestral-latest',
        timeout=60
    )

    method = MethodClass(
        llm=llm,
        llm_cluster=llm,
        profiler=ProfilerClass(log_dir=log_dir, log_style='complex', result_folder=exact_log_dir_name),
        evaluation=TaskClass,
        max_sample_nums=305,
        max_generations=31,
        pop_size=10,
        num_samplers=4,
        num_evaluators=4,
        selection_num=2,
        bounds=[LOWER_BOUND, UPPER_BOUND]  # ✅ dynamically selected
    )

    method.run()
