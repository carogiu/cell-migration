### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam

### Parameters
config = ModelParam(
    nx=500,
    ny=500,
    n=1,
    theta=1,
    epsilon=.05, # change also in model_phase
    lmbda=.01,
    #dt=5e-6,
    dt=.01,
    mob=1,
    vi="1",
    velocity=("10* sin(x[1]) + 5 * x[0] * x[0]", "0.0")
)

### Call function
main_model(config)
