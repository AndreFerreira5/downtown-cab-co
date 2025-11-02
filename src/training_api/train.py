from river import tree


model = tree.HoeffdingTreeRegressor(
    grace_period=50,
    model_selector_decay=0.3,
    seed=0
)

def run_training(*args, **kwargs):
    raise NotImplementedError("Training is not implemented yet.")
