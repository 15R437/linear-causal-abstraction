import sys
from pathlib import Path

# 1. Get the directory of uca.py (/Users/issy/Projects/linear-causal-abstraction/algorithms)
current_script_dir = Path(__file__).resolve().parent

# 2. Go up TWO levels to reach /Users/issy/Projects/
projects_root = current_script_dir.parent.parent

# 3. Point to the folder that CONTAINS the 'unsupervised_causal_reduction' package
target_project_dir = projects_root / 'targeted-causal-reduction-unsupervisedAbstraction'

sys.path.append(str(target_project_dir))
#from unsupervised_causal_reduction.causal_model.high_level import HighLevelCausalModel #type: ignore
#from unsupervised_causal_reduction.data_generators.linear import LinearCausalModel #type: ignore
#from unsupervised_causal_reduction.abstraction.causal_abstraction import CausalAbstraction #type: ignore
#from unsupervised_causal_reduction.lightning.targeted_causal_reduction import UnsupervisedCausalAbstractionModel #type: ignore

