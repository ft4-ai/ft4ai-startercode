import warnings
import logging

# Ignore deprecation warning about Lightning's use of Pytrees
warnings.filterwarnings("ignore", message=".*isinstance.*treespec, LeafSpec.*is deprecated.*") #type:ignore

# Remove Lightning ad
# See https://github.com/Lightning-AI/pytorch-lightning/issues/21294
# Moving to 2.6.5 should allow doing
#   trainer:
#      suggest_integrations: false
# Until then, we do this
class TipFilter(logging.Filter):
    def filter(self, record):
        return "💡 Tip" not in record.getMessage()
logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(TipFilter())