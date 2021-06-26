import numpy as np
from numpy import sqrt, finfo, asfarray, atleast_1d, array, zeros
import warnings

def dummy_function():
	print("script works.")

_epsilon = sqrt(finfo(float).eps)

FD_METHODS = ('2-point', '3-point', 'cs')

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