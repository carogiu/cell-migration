### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam

### Parameters
config = ModelParam(
    nx=150,
    ny=150,
    n=1000,  # number of time increments
    dim_x=20.0,  # should try to be at least 50 x 50
    dim_y=20.0,
    theta=.2,  # beta'/beta
    epsilon=.1,  # xi/l
    dt=1e-2,  # time increment
    mob=10,  # lambda in the PDF, mobility ratio M*beta # needs to be big to see the interface moving
    vi="1",  # don't change
)

### Call function
main_model(config)
