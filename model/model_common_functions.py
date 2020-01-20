import dolfin


# boundaries

class BD_right(dolfin.SubDomain):
    """
    :param dim_x: dimension in the direction of x
    """

    def __init__(self, dim_x, **kwargs):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        d_x = self.dim_x
        return on_boundary and dolfin.near(x[0], d_x / 2, 1E-12)


class BD_left(dolfin.SubDomain):
    """
    :param dim_x: dimension in the direction of x
    """

    def __init__(self, dim_x, **kwargs):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        d_x = self.dim_x
        return on_boundary and dolfin.near(x[0], -d_x / 2, 1E-12)


class BD_top_bottom(dolfin.SubDomain):
    """
    :param dim_y: dimension in the direction of y
    """

    def __init__(self, dim_y, **kwargs):
        self.dim_y = dim_y
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        d_y = self.dim_y
        return on_boundary and (dolfin.near(x[1], 0, 1E-12) or dolfin.near(x[1], d_y, 1E-12))
