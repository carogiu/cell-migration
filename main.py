### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam

### Parameters
config = ModelParam(
    nx=200,
    ny=200,
    n=8,
    dim_x=100.0,
    dim_y=50.0,
    theta=2,
    epsilon=.05,
    # dt=5e-6,
    dt=0.01,
    mob=.5,  # lambda in the PDF, mobility ratio
    vi="1", # don't change
    velocity=("10* sin(x[1]) + 5 * x[0] * x[0]", "0.0")
)

### Call function
main_model(config)
