import sys
import os

# Add the src/third_party directory to the sys.path
# I manually adjusted the imports in metric3d to support this,
# since I already had to fix some other issues there and the folder is already not a submodule because of that.
sys.path.append(os.path.dirname(__file__))

# # For MoGe, we add the third_party/MoGe directory itself and pray there are no naming conflicts,
# # since I don't want to have a separate copy of the code in this repo like I had to do with metric3d.
# sys.path.append(os.path.join(os.path.dirname(__file__), "third_party", "MoGe"))
