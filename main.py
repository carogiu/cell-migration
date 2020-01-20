### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam

### Parameters
config = ModelParam(
    nx=100,
    ny=100,
    n=50,
    dim_x=10.0,
    dim_y=10.0,
    theta=.2,
    epsilon=.02,
    dt=1e-3,
    mob=10,  # lambda in the PDF, mobility ratio M*beta # needs to be big to see the interface moving
    vi="1", # don't change
)

### Call function
main_model(config)
