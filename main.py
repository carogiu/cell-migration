### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam

### Parameters
config = ModelParam(
    nx=200,
    ny=200,
    n=3,  # number of time increments
    dim_x=50.0, # should try to be at least 50 x 50
    dim_y=50.0,
    theta=.2, # beta'/beta
    epsilon=.02, # xi/l
    dt=5e-5,  # time increment
    mob=10,  # lambda in the PDF, mobility ratio M*beta # needs to be big to see the interface moving
    vi="1", # don't change
)

### Call function
main_model(config)
