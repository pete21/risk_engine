import numpy as np

from models import utils

from . import api_pb2

NUMBER_OF_SIMULATIONS = 10000

# This model has custom model parameters
decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts two lists of doubles as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)


def returns_array(number_of_assets, asset_prices):
    """Convert flat asset price list to two-dimensional array of returns.
    :param number_of_assets: number of assets in portfolio
    :param list of floats asset_prices: flat asset price list
    :rtype ndarray
    :return: two-dimensional array that contains asset returns
    :raises ValueError: number of assets is less than 2
    :raises ValueError: number of assets prices is not a multiple of number of assets
    :raises ValueError: number of prices of individual assets is less than 2
    :raises ValueError: some asset prices are negative
    """

    if number_of_assets <= 1:
        raise ValueError('Number of assets should be greater then 1.')

    number_of_cells = len(asset_prices)

    if number_of_cells % number_of_assets != 0:
        raise ValueError('Number of asset prices should be a multiple of number of assets.')

    number_of_rows = number_of_cells // number_of_assets

    if number_of_rows < 2:
        raise ValueError('Number of price rows should be greater then 1.')

    price_array = np.array(asset_prices).reshape(number_of_rows, number_of_assets)

    if not (price_array > 0).all():
        raise ValueError('All asset prices should be positive numbers.')

    return np.diff(np.log(price_array), axis=0)


def cholesky_decomposition_of_covariance_matrix(ret_array):
    """Determine covariance matrix of returns array and decompose it to product of lower triangular
    matrix and its transpose using Cholesky decomposition

    :param ndarray ret_array: two-dimensional array that contains asset returns
    :rtype (ndarray, ndarray)
    :return: lower triangular and upper triangular matrices such that their product is equal to
        covariance matrix of returns array
    :raises ValueError: returns array contains only zeros or other values that don't imply positive
        definiteness of covariance matrix
    """

    cov_matrix = np.cov(ret_array, rowvar=False)
    try:
        lower_triangular = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        raise ValueError('All returns cannot be constant.')

    return lower_triangular, lower_triangular.T


def ldl_decomposition_of_covariance_matrix(ret_array):
    """Determine covariance matrix of returns array and decompose it to product of lower unit
    triangular matrix, diagonal matrix and upper unit triangular matrix

    :param ndarray ret_array: two-dimensional array that contains asset returns
    :rtype (ndarray, ndarray, ndarray)
    :return: lower unit triangular, diagonal and upper unit triangular matrices such that their
        product is equal to covariance matrix of returns array
    :raises ValueError: returns array contains only zeros or other values that don't imply positive
        definiteness of covariance matrix
    """

    try:
        lower_triangular, _ = cholesky_decomposition_of_covariance_matrix(ret_array)
    except ValueError:
        raise
    diagonal_vector = np.diag(lower_triangular)
    diagonal = np.diag(diagonal_vector ** 2)
    lower_triangular_normalized = lower_triangular / diagonal_vector

    return lower_triangular_normalized, diagonal, lower_triangular_normalized.T


def correlated_random_vectors_using_cholesky_decomposition(ret_array, number_of_vectors):
    """Generate given number of random vectors with mean equal to null vector and covariance
    determined by covariance matrix of given returns array. Use numpy basic random.normal function
    and multiply result by upper triangular matrix from cholesky decomposition of returns covariance
    matrix.

    :param ndarray ret_array: two-dimensional array that contains asset returns
    :param number_of_vectors: number of random vectors to be generated
    :rtype ndarray
    :return: array of random vectors possessing the same covariance matrix as given returns array
    :raises ValueError: returns array contains only zeros or other values that don't imply positive
        definiteness of covariance matrix
    """

    try:
        _, upper_triangular = cholesky_decomposition_of_covariance_matrix(ret_array)
    except ValueError:
        raise
    vector_length = upper_triangular.shape[0]
    uncorrelated_random_vectors = np.random.normal(size=(number_of_vectors, vector_length))

    return np.matrix(uncorrelated_random_vectors) * np.matrix(upper_triangular)


def correlated_random_vectors(ret_array, number_of_vectors):
    """Generate given number of random vectors with mean equal to null vector and covariance
    determined by covariance matrix of given returns array. Use numpy random.multivariate_normal
    function.

    :param ndarray ret_array: two-dimensional array that contains asset returns
    :param number_of_vectors: number of random vectors to be generated
    :rtype ndarray
    :return: array of random vectors possessing the same covariance matrix as given returns array
    """

    cov_matrix = np.cov(ret_array, rowvar=False)
    vector_length = cov_matrix.shape[0]

    return np.random.multivariate_normal(vector_length * [0.0], cov_matrix, number_of_vectors)


def calculate(parameters, data):
    """Calculate value at risk of portfolio using Monte Carlo method.

    :param parameters: model parameters
    :param data: model data
    :rtype float
    :return: VaR of portfolio
    :raises ValueError: number of assets is less than 2
    :raises ValueError: alpha parameter is non-positive or greater or equal to 100
    :raises ValueError: time perspective is negative
    :raises ValueError: length of allocation list is not equal to number of assets
    :raises ValueError: at least one asset weight is negative
    """

    number_of_assets = parameters.number_of_assets

    if number_of_assets <= 1:
        raise ValueError('Number of assets should be greater then 1.')

    alpha = parameters.alpha

    if not 0 < alpha < 100:
        raise ValueError('Parameter alpha should belong to open interval (0, 100).')

    time_perspective = parameters.time_perspective

    if time_perspective < 1:
        raise ValueError('Time perspective should be a positive integer.')

    asset_weights = np.array(data.asset_weights)

    if len(asset_weights) != number_of_assets:
        raise ValueError('Length of allocation list should be equal to number of assets.')

    if (asset_weights < 0).any():
        raise ValueError('Asset weights should be non-negative.')

    ret_array = returns_array(number_of_assets, data.asset_prices)

    random_returns = [np.sum(asset_weights * randv, axis=0) for randv in
                      (correlated_random_vectors(ret_array, time_perspective) for _ in
                       range(NUMBER_OF_SIMULATIONS))]

    var_vector = np.percentile(random_returns, number_of_assets * [alpha], axis=0,
                               overwrite_input=True)

    # var_vector = asset_weights * var_factors

    return {'result': np.sum(var_vector)}
