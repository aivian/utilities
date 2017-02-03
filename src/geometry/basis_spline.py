import numpy as np
import copy as copy

class bSpline:
    """ bSpline, a class for working with basis splines

    Contains basic routines for evaluating splines, computing the
    Vandermonde-like matrix for a spline, and computing derivatives

    John Bird (jjbird@gmail.com), AVIA lab 2014
    """

    def __init__(self,
        knots = np.asarray([0.0, 1.0]),
        order = 3,
        boundary_knots = np.vstack((np.zeros((3)), np.ones((3)))),
        coords = np.zeros((4))
        ):
        """ constructor for the bSpline class

        Args:
            knots:  the interior knots to the spline, a 1 dim numpy
                    array or array-like that can be converted
            order:  the spline order, an integer
            boundary_knots: spline boundary knots, currently they
                            cannot be coincident with each other or the
                            edge interior knots. A 2 x order numpy array
                            ([0][:] is the lower bound and [1][:] is
                            the upper)
            coords: the coordinate values defining the spline

        Returns:
            the spline object
        """

        self.interior_knots = knots
        self.boundary_knots = boundary_knots
        self.knots = np.concatenate((boundary_knots[0], knots,
            boundary_knots[1]))
        self.order = order
        self.coords = coords

    def basis(self, x):
        """ computes the spline basis

        The spline basis forms a vandermonde-like matrix which can be
        used to evaluate the spline (it is also useful in estimating
        the best-fit spline coefficients)

        y = N(x)*c

        Args:
            x:  the points on which the basis should be computed, a 1
                    dim numpy array or array-like that can be converted

        Returns:
            N:  the spline basis computed on x

        Notes:  computed using the b-spline definition given by Diercx
                in "Curve and Surface Fitting with Splines". Currently
                the algorithm as implemented can't handle coincident
                boundary knots.
        """

        N = None

        if np.ndim(x) < 1:
            x = np.asarray([x])

        for pt in x:
            E = np.zeros((self.order + len(self.interior_knots) - 1))
            for i in range(0,self.order + len(self.interior_knots) -1):
                A = 0.0
                for knt_ipj in self.knots[i:i+self.order+2]:
                    # the algorithm uses a one-sided divided difference
                    # so don't compute it if it results negative
                    if (pt <= knt_ipj):
                        B = 1.0
                        for knt_ipl in self.knots[i:i+self.order+2]:
                            if (knt_ipl != knt_ipj):
                                B *= (knt_ipj - knt_ipl)
                        A += pow(knt_ipj - pt, self.order)/B
                C = self.knots[i+self.order+1] - self.knots[i]
                E[i] = C*A
            if N is None:
                N = E
            else:
                N = np.vstack((N, E))

        return N

    def eval(self, x):
        """ evaluate a spline

        Evaluates the spline at the given points, with the currently
        defined coordinate values.

        y = N(x)*c

        Args:
            x:  the points to compute the spline value on, a 1 dim numpy
                array or array-like that can be converted

        Returns:
            y:  the value of the spline on x
        """

        N = self.basis(x)
        y = np.dot(N,self.coords)
        return y

    def __len__(self):
        """ number of coordinates required to define spline

        """
        nc = len(self.interior_knots) + self.order - 1
        return nc

class TensorProductSpline:
    """ TensorProductSpline, a class for working with tensor splines

    Contains basic routines for evaluating tensor product splines,
    computing the Vandermonde-like matrix for a spline, and computing
    derivatives.

    Based on Paul Diercx's treatment in "Curve and Surface Fittnig with
    Splines". The algorithms aren't strictly the most efficient, as this
    is optimized for usage in kalman filters.

    John Bird (jjbird@gmail.com), AVIA lab 2014
    """

    def __init__(self,
        knots = (np.asarray([0.0, 1.0]),),
        order = (1,),
        boundary_knots = (np.vstack((np.zeros((1)), np.ones((1)))),),
        coords = np.zeros((2))
        ):
        """ Constructor for the tensor product spline

        Construct the tensor product spline from spline definitions
        along each axis we're interested in. The tensor product spline
        is the cartesian product of splines in each dimension.

        Args:
            knots:  tuple, with one array for each dimension, defined as
                    for bSpline
            order:  tuple, one int for each dimension
            boundary_knots: tuple, one (2 x order) array for each axis
            coords: numpy 1 dim array with number of elements equal to
                    (order_1 + len(knots_1) - 1)*(order_2 + len(knots_2)
                    - 2)*(order_3...)... this has the size of the
                    kroneckar product of coordinates for all of the
                    splines which compose the tensor product spline

        Returns:
            tensor product spline object

        This class implements a bSpline for each dimension and handles
        the cartesian product, so it is initialized in the same way as
        bSpline, but with a tuple for each dimension.
        """

        assert type(knots) is tuple
        assert type(order) is tuple
        assert type(boundary_knots) is tuple
        assert len(knots) == len(order)
        assert len(order) == len(boundary_knots)

        spl = []
        for knts_i, k, knts_b, c in zip(
            knots, order, boundary_knots, coords):

            spl.append(bSpline(knts_i, k, knts_b))
        self.coords = coords
        self.splines = tuple(spl)
        len(self)

    def basis(self, X):
        """ computes the tensor product spline basis

        The basis of a tensor product spline is the cartesian product
        of the bases of the 1-d splines which compose it. These bases
        are computed and a kroneckar product finds the final basis.

        Args:
            x:  the points on which the basis should be computed, a
                numpy array with first dimension equal to the dimension
                of the tensor product spline and the second dimension
                full of the corresponding points (ex a 2-d spline for
                3 points is 2x3)

        Returns:
            N:  the spline basis computed on x

        Notes:  computed using the b-spline definition given by Diercx
                in "Curve and Surface Fitting with Splines". Currently
                the algorithm as implemented can't handle coincident
                boundary knots.
        """

        N = 1.0
        for S,i,x in zip(self.splines,range(0,len(self.splines)),X):
            if x.ndim == 1:
                N = np.kron(N, S.basis(x))
            else:
                N = np.kron(N, S.basis(x))
        return N

    def eval(self, x):
        """ evaluate a tensor product spline

        Evaluates the spline at the given points, with the currently
        defined coordinate values.

        if o is the kronecker product
        y = (N_1(x) o N_2(x) o ...) * c

        Args:
            x:  the points on which the spline should be evaluated, a
                numpy array with first dimension equal to the dimension
                of the tensor product spline and the second dimension
                full of the corresponding points (ex a 2-d spline for
                3 points is 2x3)

        Returns:
            y:  the value of the spline on x
        """

        N = self.basis(x)
        y = np.dot(N,self.coords)
        return y
        #if np.ndim(x) < 2:
        #    return y
        #else:
        #    return np.diagonal(np.reshape(y,(x.shape[0],x.shape[0])))



    def __len__(self):
        """ number of coordinates required to define spline

        """
        nc = 1
        for spl in self.splines:
            nc *= len(spl)
        self.shape = tuple(len(spl) for spl in self.splines)
        return nc

