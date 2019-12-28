### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam

### Parameters
config = ModelParam(
    nx=50,
    ny=50,
    n=30,
    theta=2,
    epsilon=.2,
    lmbda=.01,
    dt=5e-6,
    M=1,
    vi="1",
    velocity=("10* sin(x[1]) + 5 * x[0] * x[0]", "0.0")
)

### Call function
main_model(config)
