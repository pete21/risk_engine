import numpy as np

from math import sqrt
from scipy import optimize

from models import utils

from . import api_pb2

EPSILON = 1e-8

# This model has custom model parameters
decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts two lists of doubles and list of constraints as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)


def create_portfolio_variance(number_of_assets, variance_covariance_list):
    """Create function that returns portfolio return variance when applied to list of asset weights.

    :param int number_of_assets: number of assets in portfolio
    :param list of floats or ints variance_covariance_list: list of asset variances and covariances
        (with indices in lexicographic order: 11, 12, 13, 21, 22, 23, 31, ...)
    :rtype callable
    :return: one argument function to be applied to list of asset weights in order to get portfolio
        return variance
    :raises ValueError: number of assets is less than 2
    :raises ValueError: number of variance-covariance values is not equal to squared number of
        assets
    :raises ValueError: resulting covariance matrix is not symmetric
    :raises ValueError: resulting covariance matrix is not positive semidefinite
    """

    if number_of_assets <= 1:
        raise ValueError('Number of assets should be greater then 1.')

    if len(variance_covariance_list) != number_of_assets * number_of_assets:
        raise ValueError('Number of variance-covariance values should be equal to squared number '
                         'of assets.')

    m = np.matrix(variance_covariance_list)
    m = m.reshape((number_of_assets, number_of_assets))

    if not np.allclose(m, m.T, atol=EPSILON):
        raise ValueError('Covariance matrix populated with supplied variance-covariance values '
                         'should be symmetric.')

    if not np.all(np.linalg.eigvals(m) >= 0):
        raise ValueError('Covariance matrix populated with supplied variance-covariance values '
                         'should be positive semidefinite.')

    def quadratic_form(weights):

        if len(weights) != number_of_assets:
            raise ValueError('Number of weights should be equal to {}.'.format(number_of_assets))

        v = np.matrix(weights)
        return (v * m * v.T)[0, 0]

    return quadratic_form


def create_egain_to_sd_ratio(n, variance_covariance_list, returns_list):
    """Create function that returns negated ratio of expected gain to standard deviation of
    portfolio return when applied to list of asset weights.

    :param int n: number of assets in portfolio
    :param list of floats or ints variance_covariance_list: list of asset variances and covariances
    :param list of floats or ints returns_list: list of asset return values
    :rtype callable
    :return: one argument function to be applied to list of asset weights in order to get negated
        expected gain to SD ratio
    :raises ValueError: number of of return values is not equal to the number of assets
    """

    if len(returns_list) != n:
        raise ValueError('Number of return values should be equal to the number of assets.')

    returns = np.array(returns_list)
    portfolio_variance = create_portfolio_variance(n, variance_covariance_list)

    def ratio(weights):
        return -sum(returns * weights) / sqrt(portfolio_variance(weights))

    return ratio


def validate_constraint(a, b):
    """Perform validation of pair of constraints. If successful return unaffected pair of
    constraints, raise exception otherwise.

    :param float a: lower constraint
    :param float b: upper constraint
    :rtype (float, float)
    :return: unchanged pair of constraints
    :raises ValueError: at least one of constraints is negative or first constraint is greater then
        second constraint.
    """
    if 0 <= a <= b:
        return a, b
    else:
        raise ValueError('Constraint <{}, {}> is invalid.'.format(a, b))


def calculate(parameters, data):
    """Minimize portfolio variance or negated expected return to standard deviation ratio. Model
    parameters describe optimization type and algorithm to use.

    :param parameters: model parameters
    :param data: model data
    :rtype OptimizeResult
    :return: optimization result represented as a SciPy OptimizeResult object; important attributes
        are: x - the solution array and fun - minimal function value
    """
    number_of_assets = parameters.number_of_assets
    if parameters.type == api_pb2.ModelParams.VARIANCE:
        target_function = create_portfolio_variance(number_of_assets, data.variances)
    elif parameters.type == api_pb2.ModelParams.RETURN_TO_SD_RATIO:
        target_function = create_egain_to_sd_ratio(number_of_assets, data.variances, data.returns)
    else:
        raise ValueError('Not implemented or unknown minimization type: {}'.format(parameters.type))

    if data.additional_constraints:
        constraint_dict = {i: validate_constraint(a, b) for i, a, b in data.additional_constraints}

        if parameters.algorithm in [api_pb2.ModelParams.TRUST_CONSTR, api_pb2.ModelParams.SLSQP]:
            bound_list = list(map(lambda key: constraint_dict.get(key, (0, np.inf)),
                                  range(1, number_of_assets + 1)))
            bounds = optimize.Bounds(*zip(*bound_list))
        else:
            lower_bound_fun = lambda v: np.array(
                [x - constraint_dict.get(i + 1, (0, np.inf))[0] for i, x in enumerate(v)])
            upper_bound_fun = lambda v: np.array(
                [constraint_dict.get(i + 1, (0, np.inf))[1] - x for i, x in enumerate(v)])
            ineq_bound_cons = [{'type': 'ineq', 'fun': lower_bound_fun},
                               {'type': 'ineq', 'fun': upper_bound_fun}]
    else:
        if parameters.algorithm in [api_pb2.ModelParams.TRUST_CONSTR, api_pb2.ModelParams.SLSQP]:
            bounds = optimize.Bounds(number_of_assets * [0], number_of_assets * [np.inf])
        else:
            ineq_bound_cons = [{'type': 'ineq', 'fun': lambda v: np.array(v)}]

    if parameters.algorithm == api_pb2.ModelParams.TRUST_CONSTR:
        linear_constraints = optimize.LinearConstraint(number_of_assets * [1], 1, 1)
        result = optimize.minimize(target_function, number_of_assets * [1 / number_of_assets],
                                   method='trust-constr', bounds=bounds,
                                   constraints=linear_constraints)
    elif parameters.algorithm == api_pb2.ModelParams.SLSQP:
        eq_cons = {'type': 'eq', 'fun': lambda v: sum(v) - 1}
        result = optimize.minimize(target_function, number_of_assets * [1 / number_of_assets],
                                   method='SLSQP', bounds=bounds, constraints=eq_cons)
    elif parameters.algorithm == api_pb2.ModelParams.COBYLA:
        ineq_cons = ineq_bound_cons + [{'type': 'ineq', 'fun': lambda v: sum(v) - 1},
                                            {'type': 'ineq', 'fun': lambda v: 1 - sum(v)}]
        result = optimize.minimize(target_function, number_of_assets * [1 / number_of_assets],
                                   method='COBYLA', constraints=ineq_cons)
    else:
        raise ValueError('Not implemented or unknown minimization algorithm: {}'
                         .format(parameters.algorithm))

    return result
