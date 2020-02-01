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


def dom_and_bound(mesh, dim_x, dim_y):
    """

    :param mesh: mesh
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :return:
    """
    # define interior domain
    domain = dolfin.MeshFunction("size_t", mesh, 2) # used to define a grid cell (dimension 2)
    domain.set_all(0)
    # define the subdomains of the boundaries
    dom_left = BD_left(dim_x)
    dom_right = BD_right(dim_x)
    dom_tb = BD_top_bottom(dim_y)
    boundaries = dolfin.MeshFunction("size_t", mesh, 1)  # used to define the facets (dimension 1)
    boundaries.set_all(0)
    dom_left.mark(boundaries, 1)  # left is marked as (1)
    dom_right.mark(boundaries, 2)  # right is marked as (2)
    dom_tb.mark(boundaries, 3)  # tob-bottom is marked as (3)
    return domain, boundaries
