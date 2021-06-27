import numpy as np
from numpy import sqrt, finfo, asfarray, atleast_1d, array, zeros
import warnings

def _get_funcs(names, arrays, dtype,
               lib_name, fmodule, cmodule,
               fmodule_name, cmodule_name, alias,
               ilp64=False):
    """
    Return available BLAS/LAPACK functions.
    Used also in lapack.py. See get_blas_funcs for docstring.
    """

    funcs = []
    unpack = False
    dtype = _np.dtype(dtype)
    module1 = (cmodule, cmodule_name)
    module2 = (fmodule, fmodule_name)

    if isinstance(names, str):
        names = (names,)
        unpack = True

    prefix, dtype, prefer_fortran = find_best_blas_type(arrays, dtype)

    if prefer_fortran:
        module1, module2 = module2, module1

    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = getattr(module1[0], func_name, None)
        module_name = module1[1]
        if func is None:
            func = getattr(module2[0], func_name, None)
            module_name = module2[1]
        if func is None:
            raise ValueError(
                '%s function %s could not be found' % (lib_name, func_name))
        func.module_name, func.typecode = module_name, prefix
        func.dtype = dtype
        if not ilp64:
            func.int_dtype = _np.dtype(_np.intc)
        else:
            func.int_dtype = _np.dtype(_np.int64)
        func.prefix = prefix  # Backward compatibility
        funcs.append(func)

    if unpack:
        return funcs[0]
    else:
        return funcs

def get_blas_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available BLAS function objects from names.
    Arrays are used to determine the optimal prefix of BLAS routines.
    Parameters
    ----------
    names : str or sequence of str
        Name(s) of BLAS functions without type prefix.
    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.
    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.
    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing 'preferred' returns ILP64 routine if available,
        and otherwise the 32-bit routine. Default: False
    Returns
    -------
    funcs : list
        List containing the found function(s).
    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.
    In BLAS, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively.
    The code and the dtype are stored in attributes `typecode` and `dtype`
    of the returned functions.
    Examples
    --------
    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()
    >>> a = rng.random((3,2))
    >>> x_gemv = LA.get_blas_funcs('gemv', (a,))
    >>> x_gemv.typecode
    'd'
    >>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
    >>> x_gemv.typecode
    'z'
    """
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")

    if not ilp64:
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas, _cblas, "fblas", "cblas",
                          _blas_alias, ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError("BLAS ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas_64, None, "fblas_64", None,
                          _blas_alias, ilp64=True)

class HessianUpdateStrategy:
    """Interface for implementing Hessian update strategies.
    Many optimization methods make use of Hessian (or inverse Hessian)
    approximations, such as the quasi-Newton methods BFGS, SR1, L-BFGS.
    Some of these  approximations, however, do not actually need to store
    the entire matrix or can compute the internal matrix product with a
    given vector in a very efficiently manner. This class serves as an
    abstract interface between the optimization algorithm and the
    quasi-Newton update strategies, giving freedom of implementation
    to store and update the internal matrix as efficiently as possible.
    Different choices of initialization and update procedure will result
    in different quasi-Newton strategies.
    Four methods should be implemented in derived classes: ``initialize``,
    ``update``, ``dot`` and ``get_matrix``.
    Notes
    -----
    Any instance of a class that implements this interface,
    can be accepted by the method ``minimize`` and used by
    the compatible solvers to approximate the Hessian (or
    inverse Hessian) used by the optimization algorithms.
    """

    def initialize(self, n, approx_type):
        """Initialize internal matrix.
        Allocate internal memory for storing and updating
        the Hessian or its inverse.
        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        raise NotImplementedError("The method ``initialize(n, approx_type)``"
                                  " is not implemented.")

    def update(self, delta_x, delta_grad):
        """Update internal matrix.
        Update Hessian matrix or its inverse (depending on how 'approx_type'
        is defined) using information about the last evaluated points.
        Parameters
        ----------
        delta_x : ndarray
            The difference between two points the gradient
            function have been evaluated at: ``delta_x = x2 - x1``.
        delta_grad : ndarray
            The difference between the gradients:
            ``delta_grad = grad(x2) - grad(x1)``.
        """
        raise NotImplementedError("The method ``update(delta_x, delta_grad)``"
                                  " is not implemented.")

    def dot(self, p):
        """Compute the product of the internal matrix with the given vector.
        Parameters
        ----------
        p : array_like
            1-D array representing a vector.
        Returns
        -------
        Hp : array
            1-D represents the result of multiplying the approximation matrix
            by vector p.
        """
        raise NotImplementedError("The method ``dot(p)``"
                                  " is not implemented.")

    def get_matrix(self):
        """Return current internal matrix.
        Returns
        -------
        H : ndarray, shape (n, n)
            Dense matrix containing either the Hessian
            or its inverse (depending on how 'approx_type'
            is defined).
        """
        raise NotImplementedError("The method ``get_matrix(p)``"
                                  " is not implemented.")

class FullHessianUpdateStrategy(HessianUpdateStrategy):
    """Hessian update strategy with full dimensional internal representation.
    """
    _syr = get_blas_funcs('syr', dtype='d')  # Symmetric rank 1 update
    _syr2 = get_blas_funcs('syr2', dtype='d')  # Symmetric rank 2 update
    # Symmetric matrix-vector product
    _symv = get_blas_funcs('symv', dtype='d')

    def __init__(self, init_scale='auto'):
        self.init_scale = init_scale
        # Until initialize is called we can't really use the class,
        # so it makes sense to set everything to None.
        self.first_iteration = None
        self.approx_type = None
        self.B = None
        self.H = None

    def initialize(self, n, approx_type):
        """Initialize internal matrix.
        Allocate internal memory for storing and updating
        the Hessian or its inverse.
        Parameters
        ----------
        n : int
            Problem dimension.
        approx_type : {'hess', 'inv_hess'}
            Selects either the Hessian or the inverse Hessian.
            When set to 'hess' the Hessian will be stored and updated.
            When set to 'inv_hess' its inverse will be used instead.
        """
        self.first_iteration = True
        self.n = n
        self.approx_type = approx_type
        if approx_type not in ('hess', 'inv_hess'):
            raise ValueError("`approx_type` must be 'hess' or 'inv_hess'.")
        # Create matrix
        if self.approx_type == 'hess':
            self.B = np.eye(n, dtype=float)
        else:
            self.H = np.eye(n, dtype=float)

    def _auto_scale(self, delta_x, delta_grad):
        # Heuristic to scale matrix at first iteration.
        # Described in Nocedal and Wright "Numerical Optimization"
        # p.143 formula (6.20).
        s_norm2 = np.dot(delta_x, delta_x)
        y_norm2 = np.dot(delta_grad, delta_grad)
        ys = np.abs(np.dot(delta_grad, delta_x))
        if ys == 0.0 or y_norm2 == 0 or s_norm2 == 0:
            return 1
        if self.approx_type == 'hess':
            return y_norm2 / ys
        else:
            return ys / y_norm2

    def _update_implementation(self, delta_x, delta_grad):
        raise NotImplementedError("The method ``_update_implementation``"
                                  " is not implemented.")

    def update(self, delta_x, delta_grad):
        """Update internal matrix.
        Update Hessian matrix or its inverse (depending on how 'approx_type'
        is defined) using information about the last evaluated points.
        Parameters
        ----------
        delta_x : ndarray
            The difference between two points the gradient
            function have been evaluated at: ``delta_x = x2 - x1``.
        delta_grad : ndarray
            The difference between the gradients:
            ``delta_grad = grad(x2) - grad(x1)``.
        """
        if np.all(delta_x == 0.0):
            return
        if np.all(delta_grad == 0.0):
            warn('delta_grad == 0.0. Check if the approximated '
                 'function is linear. If the function is linear '
                 'better results can be obtained by defining the '
                 'Hessian as zero instead of using quasi-Newton '
                 'approximations.', UserWarning)
            return
        if self.first_iteration:
            # Get user specific scale
            if self.init_scale == "auto":
                scale = self._auto_scale(delta_x, delta_grad)
            else:
                scale = float(self.init_scale)
            # Scale initial matrix with ``scale * np.eye(n)``
            if self.approx_type == 'hess':
                self.B *= scale
            else:
                self.H *= scale
            self.first_iteration = False
        self._update_implementation(delta_x, delta_grad)

class BFGS(FullHessianUpdateStrategy):
    """Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.
    Parameters
    ----------
    exception_strategy : {'skip_update', 'damp_update'}, optional
        Define how to proceed when the curvature condition is violated.
        Set it to 'skip_update' to just skip the update. Or, alternatively,
        set it to 'damp_update' to interpolate between the actual BFGS
        result and the unmodified matrix. Both exceptions strategies
        are explained  in [1]_, p.536-537.
    min_curvature : float
        This number, scaled by a normalization factor, defines the
        minimum curvature ``dot(delta_grad, delta_x)`` allowed to go
        unaffected by the exception strategy. By default is equal to
        1e-8 when ``exception_strategy = 'skip_update'`` and equal
        to 0.2 when ``exception_strategy = 'damp_update'``.
    init_scale : {float, 'auto'}
        Matrix scale at first iteration. At the first
        iteration the Hessian matrix or its inverse will be initialized
        with ``init_scale*np.eye(n)``, where ``n`` is the problem dimension.
        Set it to 'auto' in order to use an automatic heuristic for choosing
        the initial scale. The heuristic is described in [1]_, p.143.
        By default uses 'auto'.
    Notes
    -----
    The update is based on the description in [1]_, p.140.
    References
    ----------
    .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """

    def __init__(self, exception_strategy='skip_update', min_curvature=None,
                 init_scale='auto'):
        if exception_strategy == 'skip_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 1e-8
        elif exception_strategy == 'damp_update':
            if min_curvature is not None:
                self.min_curvature = min_curvature
            else:
                self.min_curvature = 0.2
        else:
            raise ValueError("`exception_strategy` must be 'skip_update' "
                             "or 'damp_update'.")

        super().__init__(init_scale)
        self.exception_strategy = exception_strategy

    def _update_inverse_hessian(self, ys, Hy, yHy, s):
        """Update the inverse Hessian matrix.
        BFGS update using the formula:
            ``H <- H + ((H*y).T*y + s.T*y)/(s.T*y)^2 * (s*s.T)
                     - 1/(s.T*y) * ((H*y)*s.T + s*(H*y).T)``
        where ``s = delta_x`` and ``y = delta_grad``. This formula is
        equivalent to (6.17) in [1]_ written in a more efficient way
        for implementation.
        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
        self.H = self._syr2(-1.0 / ys, s, Hy, a=self.H)
        self.H = self._syr((ys+yHy)/ys**2, s, a=self.H)

    def _update_hessian(self, ys, Bs, sBs, y):
        """Update the Hessian matrix.
        BFGS update using the formula:
            ``B <- B - (B*s)*(B*s).T/s.T*(B*s) + y*y^T/s.T*y``
        where ``s`` is short for ``delta_x`` and ``y`` is short
        for ``delta_grad``. Formula (6.19) in [1]_.
        References
        ----------
        .. [1] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
               Second Edition (2006).
        """
        self.B = self._syr(1.0 / ys, y, a=self.B)
        self.B = self._syr(-1.0 / sBs, Bs, a=self.B)

    def _update_implementation(self, delta_x, delta_grad):
        # Auxiliary variables w and z
        if self.approx_type == 'hess':
            w = delta_x
            z = delta_grad
        else:
            w = delta_grad
            z = delta_x
        # Do some common operations
        wz = np.dot(w, z)
        Mw = self.dot(w)
        wMw = Mw.dot(w)
        # Guarantee that wMw > 0 by reinitializing matrix.
        # While this is always true in exact arithmetics,
        # indefinite matrix may appear due to roundoff errors.
        if wMw <= 0.0:
            scale = self._auto_scale(delta_x, delta_grad)
            # Reinitialize matrix
            if self.approx_type == 'hess':
                self.B = scale * np.eye(self.n, dtype=float)
            else:
                self.H = scale * np.eye(self.n, dtype=float)
            # Do common operations for new matrix
            Mw = self.dot(w)
            wMw = Mw.dot(w)
        # Check if curvature condition is violated
        if wz <= self.min_curvature * wMw:
            # If the option 'skip_update' is set
            # we just skip the update when the condion
            # is violated.
            if self.exception_strategy == 'skip_update':
                return
            # If the option 'damp_update' is set we
            # interpolate between the actual BFGS
            # result and the unmodified matrix.
            elif self.exception_strategy == 'damp_update':
                update_factor = (1-self.min_curvature) / (1 - wz/wMw)
                z = update_factor*z + (1-update_factor)*Mw
                wz = np.dot(w, z)
        # Update matrix
        if self.approx_type == 'hess':
            self._update_hessian(wz, Mw, wMw, z)
        else:
            self._update_inverse_hessian(wz, Mw, wMw, z)

class NonlinearConstraint:
    """Nonlinear constraint on the variables.
    The constraint has the general inequality form::
        lb <= fun(x) <= ub
    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and ``fun`` returns a vector with m components.
    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.
    Parameters
    ----------
    fun : callable
        The function defining the constraint.
        The signature is ``fun(x) -> array_like, shape (m,)``.
    lb, ub : array_like
        Lower and upper bounds on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary.
    jac : {callable,  '2-point', '3-point', 'cs'}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix,
        where element (i, j) is the partial derivative of f[i] with
        respect to x[j]).  The keywords {'2-point', '3-point',
        'cs'} select a finite difference scheme for the numerical estimation.
        A callable must have the following signature:
        ``jac(x) -> {ndarray, sparse matrix}, shape (m, n)``.
        Default is '2-point'.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy, None}, optional
        Method for computing the Hessian matrix. The keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme for
        numerical  estimation.  Alternatively, objects implementing
        `HessianUpdateStrategy` interface can be used to approximate the
        Hessian. Currently available implementations are:
            - `BFGS` (default option)
            - `SR1`
        A callable must return the Hessian matrix of ``dot(fun, v)`` and
        must have the following signature:
        ``hess(x, v) -> {LinearOperator, sparse matrix, array_like}, shape (n, n)``.
        Here ``v`` is ndarray with shape (m,) containing Lagrange multipliers.
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    finite_diff_rel_step: None or array_like, optional
        Relative step size for the finite difference approximation. Default is
        None, which will select a reasonable value automatically depending
        on a finite difference scheme.
    finite_diff_jac_sparsity: {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        difference estimation, its shape must be (m, n). If the Jacobian has
        only few non-zero elements in *each* row, providing the sparsity
        structure will greatly speed up the computations. A zero entry means
        that a corresponding element in the Jacobian is identically zero.
        If provided, forces the use of 'lsmr' trust-region solver.
        If None (default) then dense differencing will be used.
    Notes
    -----
    Finite difference schemes {'2-point', '3-point', 'cs'} may be used for
    approximating either the Jacobian or the Hessian. We, however, do not allow
    its use for approximating both simultaneously. Hence whenever the Jacobian
    is estimated via finite-differences, we require the Hessian to be estimated
    using one of the quasi-Newton strategies.
    The scheme 'cs' is potentially the most accurate, but requires the function
    to correctly handles complex inputs and be analytically continuable to the
    complex plane. The scheme '3-point' is more accurate than '2-point' but
    requires twice as many operations.
    Examples
    --------
    Constrain ``x[0] < sin(x[1]) + 1.9``
    >>> from scipy.optimize import NonlinearConstraint
    >>> con = lambda x: x[0] - np.sin(x[1])
    >>> nlc = NonlinearConstraint(con, -np.inf, 1.9)
    """
    def __init__(self, fun, lb, ub, jac='2-point', hess=BFGS(),
                 keep_feasible=False, finite_diff_rel_step=None,
                 finite_diff_jac_sparsity=None):
        self.fun = fun
        self.lb = lb
        self.ub = ub
        self.finite_diff_rel_step = finite_diff_rel_step
        self.finite_diff_jac_sparsity = finite_diff_jac_sparsity
        self.jac = jac
        self.hess = hess
        self.keep_feasible = keep_feasible

def dummy_function():
    print("script works.")

_epsilon = sqrt(finfo(float).eps)

FD_METHODS = ('2-point', '3-point', 'cs')

def minimize_hack(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)
    if not isinstance(args, tuple):
        args = (args,)     
    if method is None:
        # Select automatically
        if constraints:
            method = 'SLSQP'
        elif bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'
    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()
    if options is None:
        options = {}
    # check if optional parameters are supported by the selected method
    # - jac
    if meth in ('nelder-mead', 'powell', 'cobyla') and bool(jac):
        warn('Method %s does not use gradient information (jac).' % method,
             RuntimeWarning)               
    # - hess
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', 'trust-exact', '_custom') and hess is not None:
        warn('Method %s does not use Hessian information (hess).' % method,
             RuntimeWarning) 
    # - hessp
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', '_custom') \
       and hessp is not None:
        warn('Method %s does not use Hessian-vector product '
             'information (hessp).' % method, RuntimeWarning)
    # - constraints or bounds
    if (meth in ('cg', 'bfgs', 'newton-cg', 'dogleg', 'trust-ncg')
            and (bounds is not None or np.any(constraints))):
        warn('Method %s cannot handle constraints nor bounds.' % method,
             RuntimeWarning)
    if meth in ('nelder-mead', 'l-bfgs-b', 'tnc', 'powell') and np.any(constraints):
        warn('Method %s cannot handle constraints.' % method,
             RuntimeWarning)
    if meth == 'cobyla' and bounds is not None:
        warn('Method %s cannot handle bounds.' % method,
             RuntimeWarning)
    # - callback
    if (meth in ('cobyla',) and callback is not None):
        warn('Method %s does not support callback.' % method, RuntimeWarning)
    # - return_all
    if (meth in ('l-bfgs-b', 'tnc', 'cobyla', 'slsqp') and
            options.get('return_all', False)):
        warn('Method %s does not support the return_all option.' % method,
             RuntimeWarning)        
    # check gradient vector
    if callable(jac):
        pass
    elif jac is True:
        # fun returns func and grad
        fun = MemoizeJac(fun)
        jac = fun.derivative
    elif (jac in FD_METHODS and
          meth in ['trust-constr', 'bfgs', 'cg', 'l-bfgs-b', 'tnc', 'slsqp']):
        # finite differences with relative step
        pass
    elif meth in ['trust-constr']:
        # default jac calculation for this method
        jac = '2-point'
    elif jac is None or bool(jac) is False:
        # this will cause e.g. LBFGS to use forward difference, absolute step
        jac = None
    else:
        # default if jac option is not understood
        jac = None   
    # set default tolerances
    if tol is not None:
        options = dict(options)
        if meth == 'nelder-mead':
            options.setdefault('xatol', tol)
            options.setdefault('fatol', tol)
        if meth in ('newton-cg', 'powell', 'tnc'):
            options.setdefault('xtol', tol)
        if meth in ('powell', 'l-bfgs-b', 'tnc', 'slsqp'):
            options.setdefault('ftol', tol)
        if meth in ('bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg',
                    'trust-ncg', 'trust-exact', 'trust-krylov'):
            options.setdefault('gtol', tol)
        if meth in ('cobyla', '_custom'):
            options.setdefault('tol', tol)
        if meth == 'trust-constr':
            options.setdefault('xtol', tol)
            options.setdefault('gtol', tol)
            options.setdefault('barrier_tol', tol)

    if meth == '_custom':
        # custom method called before bounds and constraints are 'standardised'
        # custom method should be able to accept whatever bounds/constraints
        # are provided to it.
        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                      bounds=bounds, constraints=constraints,
                      callback=callback, **options)

    if bounds is not None:
        bounds = standardize_bounds(bounds, x0, meth)

    if constraints is not None:
        constraints = standardize_constraints(constraints, x0, meth)         

    if meth == 'slsqp':
        return _minimize_slsqp(fun, x0, args, jac, bounds, constraints, callback=callback, **options)
            
    print("Is it here?")                            

class OptimizeWarning(UserWarning):
    pass

class ScalarFunction:
    """Scalar function and its derivatives.
    This class defines a scalar function F: R^n->R and methods for
    computing or approximating its first and second derivatives.
    Parameters
    ----------
    fun : callable
        evaluates the scalar function. Must be of the form ``fun(x, *args)``,
        where ``x`` is the argument in the form of a 1-D array and ``args`` is
        a tuple of any additional fixed parameters needed to completely specify
        the function. Should return a scalar.
    x0 : array-like
        Provides an initial set of variables for evaluating fun. Array of real
        elements of size (n,), where 'n' is the number of independent
        variables.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the scalar
        function.
    grad : {callable, '2-point', '3-point', 'cs'}
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:
            ``grad(x, *args) -> array_like, shape (n,)``
        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}
        Method for computing the Hessian matrix. If it is callable, it should
        return the  Hessian matrix:
            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
        where x is a (n,) ndarray and `args` is a tuple with the fixed
        parameters. Alternatively, the keywords {'2-point', '3-point', 'cs'}
        select a finite difference scheme for numerical estimation. Or, objects
        implementing `HessianUpdateStrategy` interface can be used to
        approximate the Hessian.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.
    finite_diff_rel_step : None or array_like
        Relative step size to use. The absolute step size is computed as
        ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``, possibly
        adjusted to fit into the bounds. For ``method='3-point'`` the sign
        of `h` is ignored. If None then finite_diff_rel_step is selected
        automatically,
    finite_diff_bounds : tuple of array_like
        Lower and upper bounds on independent variables. Defaults to no bounds,
        (-np.inf, np.inf). Each bound must match the size of `x0` or be a
        scalar, in the latter case the bound will be the same for all
        variables. Use it to limit the range of function evaluation.
    epsilon : None or array_like, optional
        Absolute step size to use, possibly adjusted to fit into the bounds.
        For ``method='3-point'`` the sign of `epsilon` is ignored. By default
        relative steps are used, only if ``epsilon is not None`` are absolute
        steps used.
    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `grad`, hess` and corresponding attributes `f`, `g` and `H`. The following
    things should be considered:
        1. Use only public methods `fun`, `grad` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """
    def __init__(self, fun, x0, args, grad, hess, finite_diff_rel_step,
                 finite_diff_bounds, epsilon=None):
        if not callable(grad) and grad not in FD_METHODS:
            raise ValueError(
                f"`grad` must be either callable or one of {FD_METHODS}."
            )

        if not (callable(hess) or hess in FD_METHODS
                or isinstance(hess, HessianUpdateStrategy)):
            raise ValueError(
                f"`hess` must be either callable, HessianUpdateStrategy"
                f" or one of {FD_METHODS}."
            )

        if grad in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the gradient is estimated via "
                             "finite-differences, we require the Hessian "
                             "to be estimated using one of the "
                             "quasi-Newton strategies.")

        # the astype call ensures that self.x is a copy of x0
        self.x = np.atleast_1d(x0).astype(float)
        self.n = self.x.size
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False

        finite_diff_options = {}
        if grad in FD_METHODS:
            finite_diff_options["method"] = grad
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["bounds"] = finite_diff_bounds
        if hess in FD_METHODS:
            finite_diff_options["method"] = hess
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["as_linear_operator"] = True

        # Function evaluation
        def fun_wrapped(x):
            self.nfev += 1
            # Send a copy because the user may overwrite it.
            # Overwriting results in undefined behaviour because
            # fun(self.x) will change self.x, with the two no longer linked.
            fx = fun(np.copy(x), *args)
            # Make sure the function returns a true scalar
            if not np.isscalar(fx):
                try:
                    fx = np.asarray(fx).item()
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        "The user-provided objective function "
                        "must return a scalar value."
                    ) from e
            return fx

        def update_fun():
            self.f = fun_wrapped(self.x)

        self._update_fun_impl = update_fun
        self._update_fun()

        # Gradient evaluation
        if callable(grad):
            def grad_wrapped(x):
                self.ngev += 1
                return np.atleast_1d(grad(np.copy(x), *args))

            def update_grad():
                self.g = grad_wrapped(self.x)

        elif grad in FD_METHODS:
            def update_grad():
                self._update_fun()
                self.ngev += 1
                self.g = approx_derivative(fun_wrapped, self.x, f0=self.f,
                                           **finite_diff_options)

        self._update_grad_impl = update_grad
        self._update_grad()

        # Hessian Evaluation
        if callable(hess):
            self.H = hess(np.copy(x0), *args)
            self.H_updated = True
            self.nhev += 1

            if sps.issparse(self.H):
                def hess_wrapped(x):
                    self.nhev += 1
                    return sps.csr_matrix(hess(np.copy(x), *args))
                self.H = sps.csr_matrix(self.H)

            elif isinstance(self.H, LinearOperator):
                def hess_wrapped(x):
                    self.nhev += 1
                    return hess(np.copy(x), *args)

            else:
                def hess_wrapped(x):
                    self.nhev += 1
                    return np.atleast_2d(np.asarray(hess(np.copy(x), *args)))
                self.H = np.atleast_2d(np.asarray(self.H))

            def update_hess():
                self.H = hess_wrapped(self.x)

        elif hess in FD_METHODS:
            def update_hess():
                self._update_grad()
                self.H = approx_derivative(grad_wrapped, self.x, f0=self.g,
                                           **finite_diff_options)
                return self.H

            update_hess()
            self.H_updated = True
        elif isinstance(hess, HessianUpdateStrategy):
            self.H = hess
            self.H.initialize(self.n, 'hess')
            self.H_updated = True
            self.x_prev = None
            self.g_prev = None

            def update_hess():
                self._update_grad()
                self.H.update(self.x - self.x_prev, self.g - self.g_prev)

        self._update_hess_impl = update_hess

        if isinstance(hess, HessianUpdateStrategy):
            def update_x(x):
                self._update_grad()
                self.x_prev = self.x
                self.g_prev = self.g
                # ensure that self.x is a copy of x. Don't store a reference
                # otherwise the memoization doesn't work properly.
                self.x = np.atleast_1d(x).astype(float)
                self.f_updated = False
                self.g_updated = False
                self.H_updated = False
                self._update_hess()
        else:
            def update_x(x):
                # ensure that self.x is a copy of x. Don't store a reference
                # otherwise the memoization doesn't work properly.
                self.x = np.atleast_1d(x).astype(float)
                self.f_updated = False
                self.g_updated = False
                self.H_updated = False
        self._update_x_impl = update_x

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_grad(self):
        if not self.g_updated:
            self._update_grad_impl()
            self.g_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    def fun(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_fun()
        return self.f

    def grad(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_grad()
        return self.g

    def hess(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_hess()
        return self.H

    def fun_and_grad(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_fun()
        self._update_grad()
        return self.f, self.g

def standardize_constraints(constraints, x0, meth):
    """Converts constraints to the form required by the solver."""
    all_constraint_types = (NonlinearConstraint, LinearConstraint, dict)
    new_constraint_types = all_constraint_types[:-1]
    if isinstance(constraints, all_constraint_types):
        constraints = [constraints]
    constraints = list(constraints)  # ensure it's a mutable sequence

    if meth == 'trust-constr':
        for i, con in enumerate(constraints):
            if not isinstance(con, new_constraint_types):
                constraints[i] = old_constraint_to_new(i, con)
    else:
        # iterate over copy, changing original
        for i, con in enumerate(list(constraints)):
            if isinstance(con, new_constraint_types):
                old_constraints = new_constraint_to_old(con, x0)
                constraints[i] = old_constraints[0]
                constraints.extend(old_constraints[1:])  # appends 1 if present

    return constraints

def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """
    Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:
            ``jac(x, *args) -> array_like, shape (n,)``
        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    hess : {callable,  '2-point', '3-point', 'cs', None}
        Computes the Hessian matrix. If it is callable, it should return the
        Hessian matrix:
            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
        Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
        finite difference scheme for numerical estimation.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.
    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # ScalarFunction requires something for hess, so we give a dummy
        # implementation here if nothing is provided, return a value of None
        # so that downstream minimisers halt. The results of `fun.hess`
        # should not be used.
        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

def _minimize_slsqp(func, x0, args=(), jac=None, bounds=None,
                    constraints=(),
                    maxiter=100, ftol=1.0E-6, iprint=1, disp=False,
                    eps=_epsilon, callback=None, finite_diff_rel_step=None,
                    **unknown_options):
    """
    Minimize a scalar function of one or more variables using Sequential
    Least Squares Programming (SLSQP).
    Options
    -------
    ftol : float
        Precision goal for the value of f in the stopping criterion.
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored and set to 0.
    maxiter : int
        Maximum number of iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of `jac`. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    _check_unknown_options(unknown_options)
    iter = maxiter - 1
    acc = ftol
    epsilon = eps

    if not disp:
        iprint = 0

    # Transform x0 into an array.
    x = asfarray(x0).flatten()

    # SLSQP is sent 'old-style' bounds, 'new-style' bounds are required by
    # ScalarFunction
    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        new_bounds = old_bound_to_new(bounds)

    # clip the initial guess to bounds, otherwise ScalarFunction doesn't work
    x = np.clip(x, new_bounds[0], new_bounds[1])

    # Constraints are triaged per type into a dictionary of tuples
    if isinstance(constraints, dict):
        constraints = (constraints, )

    cons = {'eq': (), 'ineq': ()}
    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])

        # check function
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # check Jacobian
        cjac = con.get('jac')
        if cjac is None:
            # approximate Jacobian function. The factory function is needed
            # to keep a reference to `fun`, see gh-4240.
            def cjac_factory(fun):
                def cjac(x, *args):
                    x = _check_clip_x(x, new_bounds)

                    if jac in ['2-point', '3-point', 'cs']:
                        return approx_derivative(fun, x, method=jac, args=args,
                                                 rel_step=finite_diff_rel_step,
                                                 bounds=new_bounds)
                    else:
                        return approx_derivative(fun, x, method='2-point',
                                                 abs_step=epsilon, args=args,
                                                 bounds=new_bounds)

                return cjac
            cjac = cjac_factory(con['fun'])

        # update constraints' dictionary
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())}, )

    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit reached"}

    # Set the parameters that SLSQP will need
    # meq, mieq: number of equality and inequality constraints
    meq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
              for c in cons['eq']]))
    mieq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
               for c in cons['ineq']]))
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = array([1, m]).max()
    # n = The number of independent variables
    n = len(x)

    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w = zeros(len_w)
    jw = zeros(len_jw)

    # Decompose bounds into xl and xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = array([(_arr_to_scalar(l), _arr_to_scalar(u))
                      for (l, u) in bounds], float)
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))
        xl, xu = bnds[:, 0], bnds[:, 1]

        # Mark infinite bounds with nans; the Fortran code understands this
        infbnd = ~isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    # ScalarFunction provides function and gradient evaluation
    sf = _prepare_scalar_function(func, x, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  bounds=new_bounds)
    # gh11403 SLSQP sometimes exceeds bounds by 1 or 2 ULP, make sure this
    # doesn't get sent to the func/grad evaluator.
    wrapped_fun = _clip_x_for_func(sf.fun, new_bounds)
    wrapped_grad = _clip_x_for_func(sf.grad, new_bounds)

    # Initialize the iteration counter and the mode value
    mode = array(0, int)
    acc = array(acc, float)
    majiter = array(iter, int)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    alpha = array(0, float)
    f0 = array(0, float)
    gs = array(0, float)
    h1 = array(0, float)
    h2 = array(0, float)
    h3 = array(0, float)
    h4 = array(0, float)
    t = array(0, float)
    t0 = array(0, float)
    tol = array(0, float)
    iexact = array(0, int)
    incons = array(0, int)
    ireset = array(0, int)
    itermx = array(0, int)
    line = array(0, int)
    n1 = array(0, int)
    n2 = array(0, int)
    n3 = array(0, int)

    # Print the header if iprint >= 2
    if iprint >= 2:
        print("%5s %5s %16s %16s" % ("NIT", "FC", "OBJFUN", "GNORM"))

    # mode is zero on entry, so call objective, constraints and gradients
    # there should be no func evaluations here because it's cached from
    # ScalarFunction
    fx = wrapped_fun(x)
    g = append(wrapped_grad(x), 0.0)
    c = _eval_constraint(x, cons)
    a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

    while 1:
        # Call SLSQP
        slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line,
              n1, n2, n3)

        if mode == 1:  # objective and constraint evaluation required
            fx = wrapped_fun(x)
            c = _eval_constraint(x, cons)

        if mode == -1:  # gradient evaluation required
            g = append(wrapped_grad(x), 0.0)
            a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

        if majiter > majiter_prev:
            # call callback if major iteration has incremented
            if callback is not None:
                callback(np.copy(x))

            # Print the status of the current iterate if iprint > 2
            if iprint >= 2:
                print("%5i %5i % 16.6E % 16.6E" % (majiter, sf.nfev,
                                                   fx, linalg.norm(g)))

        # If exit mode is not -1 or 1, slsqp has completed
        if abs(mode) != 1:
            break

        majiter_prev = int(majiter)

    # Optimization loop complete. Print status if requested
    if iprint >= 1:
        print(exit_modes[int(mode)] + "    (Exit mode " + str(mode) + ')')
        print("            Current function value:", fx)
        print("            Iterations:", majiter)
        print("            Function evaluations:", sf.nfev)
        print("            Gradient evaluations:", sf.ngev)

    return OptimizeResult(x=x, fun=fx, jac=g[:-1], nit=int(majiter),
                          nfev=sf.nfev, njev=sf.ngev, status=int(mode),
                          message=exit_modes[int(mode)], success=(mode == 0))


'''
    """Minimization of scalar function of one or more variables.
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    method : str or callable, optional
        Type of solver.  Should be one of
            - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
            - custom - a callable object (added in version 0.14.0),
              see below for description.
        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector. Only for CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
        trust-exact and trust-constr.
        If it is a callable, it should be a function that returns the gradient
        vector:
            ``jac(x, *args) -> array_like, shape (n,)``
        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters. If `jac` is a Boolean and is True, `fun` is
        assumed to return a tuple ``(f, g)`` containing the objective
        function and the gradient.
        Methods 'Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact', and
        'trust-krylov' require that either a callable be supplied, or that
        `fun` return the objective and gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
        Method for computing the Hessian matrix. Only for Newton-CG, dogleg,
        trust-ncg, trust-krylov, trust-exact and trust-constr. If it is
        callable, it should return the Hessian matrix:
            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
        where x is a (n,) ndarray and `args` is a tuple with the fixed
        parameters. LinearOperator and sparse matrix returns are only allowed
        for 'trust-constr' method. Alternatively, the keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme
        for numerical estimation. Or, objects implementing the
        `HessianUpdateStrategy` interface can be used to approximate
        the Hessian. Available quasi-Newton methods implementing
        this interface are:
            - `BFGS`;
            - `SR1`.
        Whenever the gradient is estimated via finite-differences,
        the Hessian cannot be estimated with options
        {'2-point', '3-point', 'cs'} and needs to be
        estimated using one of the quasi-Newton strategies.
        'trust-exact' cannot use a finite-difference scheme, and must be used
        with a callable returning an (n, n) array.
    hessp : callable, optional
        Hessian of objective function times an arbitrary vector p. Only for
        Newton-CG, trust-ncg, trust-krylov, trust-constr.
        Only one of `hessp` or `hess` needs to be given.  If `hess` is
        provided, then `hessp` will be ignored.  `hessp` must compute the
        Hessian times an arbitrary vector:
            ``hessp(x, p, *args) ->  ndarray shape (n,)``
        where x is a (n,) ndarray, p is an arbitrary vector with
        dimension (n,) and `args` is a tuple with the fixed
        parameters.
    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and
        trust-constr methods. There are two ways to specify the bounds:
            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition (only for COBYLA, SLSQP and trust-constr).
        Constraints for 'trust-constr' are defined as a single object or a
        list of objects specifying constraints to the optimization problem.
        Available constraints are:
            - `LinearConstraint`
            - `NonlinearConstraint`
        Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
        Each dictionary with fields:
            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.
        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
        Note that COBYLA only supports inequality constraints.
    tol : float, optional
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:
            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see :func:`show_options()`.
    callback : callable, optional
        Called after each iteration. For 'trust-constr' it is a callable with
        the signature:
            ``callback(xk, OptimizeResult state) -> bool``
        where ``xk`` is the current parameter vector. and ``state``
        is an `OptimizeResult` object, with the same fields
        as the ones from the return. If callback returns True
        the algorithm execution is terminated.
        For all the other methods, the signature is:
            ``callback(xk)``
        where ``xk`` is the current parameter vector.
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential
    Least SQuares Programming to minimize a function of several
    variables with any combination of bounds, equality and inequality
    constraints. The method wraps the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft [12]_. Note that the
    wrapper handles infinite values in bounds by converting them into
    large floating values.
'''