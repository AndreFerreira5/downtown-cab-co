from river import tree, metrics as river_metrics
from src.training_api.data.loader import DataLoader

data_loader = DataLoader()

model = tree.HoeffdingAdaptiveTreeRegressor(
    grace_period=50,
    model_selector_decay=0.3,
    seed=0
)


def train_hatr(df, target_col, model_params):
    model = tree.HoeffdingAdaptiveTreeRegressor()
    mae = river_metrics.MAE()
    rmse = river_metrics.RMSE()
    f1 = river_metrics.F1()


def run_training(*args, **kwargs):
    pass
