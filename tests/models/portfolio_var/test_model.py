from collections import namedtuple
from math import sqrt
from unittest import TestCase, main

import numpy as np
from scipy import stats

from models.portfolio_var.model import returns_array, cholesky_decomposition_of_covariance_matrix,\
    ldl_decomposition_of_covariance_matrix, correlated_random_vectors_using_cholesky_decomposition,\
    correlated_random_vectors, calculate

MOCK_RETURN_ARRAY = np.array([[-3.23752306e-03, 1.20040814e-04, 5.01871230e-03],
                              [3.15016011e-03, -1.15298708e-03, 1.41582920e-03],
                              [-6.13445093e-03, 3.12398623e-04, -7.19427563e-03],
                              [-5.81858293e-03, 3.62149788e-03, -6.65241174e-04],
                              [-4.42184398e-04, 7.25142123e-03, -4.86016109e-03],
                              [8.84173356e-04, 5.90056097e-03, -9.21311699e-03],
                              [2.55969070e-03, 9.68248609e-04, -1.26129546e-02],
                              [-3.35541154e-03, -1.53546360e-03, -7.05470298e-03],
                              [-5.49890521e-03, -3.38634185e-03, -1.54580962e-02],
                              [-4.09946212e-03, 2.88976467e-03, -1.99704065e-02],
                              [3.68204286e-02, 1.68398369e-02, 1.28550786e-02],
                              [-2.15415188e-03, 1.41769822e-03, 5.41626199e-03],
                              [-1.20835506e-03, 5.02705702e-03, 8.96241809e-03],
                              [2.24292710e-03, -2.56828656e-03, 3.26619779e-03],
                              [-1.46595982e-03, 3.47451447e-04, 7.90201543e-04],
                              [1.03022889e-02, -3.96812105e-03, 1.48963946e-02],
                              [5.70627107e-03, -3.37716197e-03, 1.55539376e-01],
                              [5.94303197e-04, -9.10289838e-04, 0.00000000e+00],
                              [1.86551398e-03, -1.37868206e-03, 0.00000000e+00],
                              [7.62162886e-04, -1.54451028e-03, 0.00000000e+00],
                              [-3.81663667e-03, 4.74294414e-03, 0.00000000e+00],
                              [5.33922197e-03, 4.79014984e-03, 8.32604805e-05],
                              [6.99097393e-03, -7.65812321e-04, 2.49739856e-04],
                              [8.39313442e-05, -2.23115811e-03, 1.66458594e-04],
                              [1.06854677e-02, 1.72029055e-03, 4.99209595e-04],
                              [8.10457628e-03, -7.48386077e-03, 1.66347834e-04],
                              [1.56372198e-03, 8.64454617e-03, 3.32612676e-04],
                              [3.03814332e-03, -1.85774384e-03, 0.00000000e+00],
                              [1.80209747e-03, 1.27756954e-03, -2.49449134e-04],
                              [-4.91159145e-04, -7.29241492e-03, 5.81951216e-04],
                              [3.75940292e-03, -2.34109904e-03, 2.49304028e-04],
                              [1.63012507e-03, 1.70954224e-03, 3.32308718e-04],
                              [4.88519795e-04, -7.42135863e-03, 1.07924132e-03],
                              [1.31809615e-02, -3.87341061e-03, -3.49098518e-03],
                              [7.12430470e-03, -4.67273944e-03, -8.32674133e-05],
                              [-8.89322049e-03, -6.65532077e-03, 1.66527894e-04],
                              [1.92957127e-03, -1.12551947e-03, -8.32604805e-05],
                              [-1.76863138e-03, 1.50838619e-03, 1.66514029e-04],
                              [-2.90088842e-03, -4.29169773e-03, 9.98502330e-04],
                              [-1.85776072e-03, -4.08550732e-04, -2.49532129e-04],
                              [-9.01455733e-03, 2.16312356e-04, 1.66361671e-04],
                              [8.44846617e-03, 4.08462385e-04, -1.66361671e-04],
                              [-4.13407957e-03, -3.36368663e-04, 1.16395091e-03],
                              [1.62324522e-03, -3.84559924e-04, -2.49304028e-04],
                              [7.51366896e-03, 9.37105853e-04, 6.64672673e-04],
                              [3.61547817e-03, 3.47642974e-03, -9.97174754e-04],
                              [1.12215465e-03, 1.36329412e-03, -8.31739215e-04],
                              [-2.40355728e-04, -9.80427401e-04, 3.32778705e-04],
                              [-4.00721304e-04, 7.17711935e-05, 3.32668000e-04],
                              [-4.09655579e-03, 1.60154930e-03, 3.32557369e-04],
                              [-1.12748663e-03, 1.78974168e-03, -2.49407658e-04],
                              [9.66494921e-04, 2.47648656e-03, -2.49469878e-04],
                              [9.37469307e-03, 2.20936814e-03, -8.32016023e-04],
                              [-3.19055598e-04, 1.04359385e-03, 0.00000000e+00],
                              [-1.67671403e-03, -1.01986371e-03, -8.32396887e-05],
                              [-1.43942448e-03, 2.29915128e-03, 0.00000000e+00],
                              [-4.81310090e-03, 1.34858596e-03, 1.66472449e-04],
                              [1.60694234e-03, -3.07412183e-04, 2.49656723e-04],
                              [8.82718831e-04, -2.13083396e-03, 4.15956081e-04],
                              [-3.45506752e-03, 1.01863157e-03, 8.31704579e-05],
                              [4.97793079e-03, -2.08575364e-03, 3.32612676e-04],
                              [-7.47621956e-03, 3.24525637e-03, 9.97174754e-04],
                              [9.95592326e-03, -1.65684433e-03, 4.15196186e-04],
                              [-2.96035740e-03, 2.88586835e-03, -1.66057788e-04],
                              [2.72065463e-03, -2.08077253e-03, 9.95933355e-04],
                              [-1.67953013e-03, -1.42119499e-03, -8.29565723e-05],
                              [2.47831601e-03, -2.60771648e-04, 1.07789903e-03],
                              [5.89126774e-03, -2.13409213e-04, -6.63184970e-04],
                              [1.09730724e-02, 1.99005041e-03, 0.00000000e+00],
                              [8.63252947e-04, 1.56083749e-03, -2.48807797e-04],
                              [-5.42731501e-03, -1.93958951e-03, 3.31729975e-04],
                              [-1.57865689e-03, -2.36767648e-05, 0.00000000e+00],
                              [7.89639965e-04, 1.30139030e-03, 5.80262792e-04],
                              [1.89259579e-03, -1.04097672e-03, -4.14439063e-04],
                              [5.42134504e-03, -1.46867254e-03, 4.97306268e-04],
                              [8.61562614e-04, 1.01882465e-03, -3.31510031e-04],
                              [3.90686542e-03, 2.15270058e-03, 8.28569109e-04],
                              [-5.78759632e-03, -7.56465451e-04, -1.07727377e-03],
                              [-6.53159906e-03, -4.85972960e-03, 9.11614864e-04],
                              [-1.58027846e-03, -1.16509946e-03, 5.79686158e-04],
                              [2.60612191e-03, -2.76361733e-03, -1.65590330e-04],
                              [-3.23893319e-03, -1.93430587e-03, 1.24125962e-03],
                              [9.84221165e-03, -1.84226004e-03, 1.23972081e-03],
                              [-9.28847988e-03, 3.84813506e-03, 2.06279210e-03],
                              [-3.00966487e-03, -1.88636501e-03, -1.40223556e-03],
                              [-3.33704425e-03, 9.55977248e-05, 4.12626373e-04],
                              [4.05067867e-03, -1.31527048e-03, -2.56103242e-03],
                              [1.18826013e-03, -6.70257356e-04, -3.30934064e-04],
                              [-2.21922894e-03, 1.29223719e-03, -1.07629269e-03],
                              [-1.58818425e-03, -9.33114877e-04, -7.45804882e-04],
                              [9.33403391e-03, 1.22005206e-03, 6.62965136e-04],
                              [2.43777890e-03, -5.26114418e-04, 0.00000000e+00],
                              [-1.57208019e-03, -8.13630752e-04, 4.14130126e-04],
                              [8.92793149e-03, -1.82110131e-03, 8.28054486e-05],
                              [5.13260323e-03, 1.41403755e-03, -6.62635657e-04],
                              [-3.65233339e-03, 1.48378632e-03, 0.00000000e+00],
                              [5.44810692e-04, 1.45770879e-03, -1.65727544e-04],
                              [1.55496844e-03, 1.26481098e-03, 1.57330385e-03]])

MOCK_ASSET_PRICES = [1.1189, 4.1911, 1.0413,
                     1.1133, 4.1893, 1.0376,
                     1.1202, 4.1757, 1.0447,
                     1.1206, 4.1727, 1.0507,
                     1.1213, 4.1671, 1.0449,
                     1.1204, 4.1697, 1.0449,
                     1.1345, 4.1625, 1.0435,
                     1.1299, 4.1729, 1.0449,
                     1.1404, 4.1667, 1.0461,
                     1.1279, 4.1576, 1.0451,
                     1.1215, 4.1448, 1.0465,
                     1.1218, 4.1573, 1.0512,
                     1.122, 4.1497, 1.0469,
                     1.1232, 4.1405, 1.0523,
                     1.1279, 4.15, 1.0486,
                     1.1249, 4.1657, 1.0465,
                     1.1162, 4.1551, 1.047,
                     1.1218, 4.1441, 1.0478,
                     1.1317, 4.1692, 1.0551,
                     1.1134, 4.1261, 1.0427,
                     1.1029, 4.1288, 1.0391,
                     1.0944, 4.1335, 1.0331,
                     1.097, 4.1298, 1.0341,
                     1.0896, 4.1289, 1.0344,
                     1.0863, 4.1335, 1.0331,
                     1.0926, 4.1391, 1.0351,
                     1.0978, 4.1075, 1.0349,
                     1.1164, 4.1003, 1.0417,
                     1.1133, 4.0959, 1.0394,
                     1.1118, 4.0653, 1.0429,
                     1.118, 4.0486, 1.042,
                     1.1389, 4.0615, 1.0482,
                     1.1328, 4.0432, 1.0463,
                     1.1419, 4.0827, 1.0386,
                     1.1221, 4.0872, 1.041,
                     1.1239, 4.1028, 1.0396,
                     1.1142, 4.0668, 1.0403,
                     1.1221, 4.0679, 1.0385,
                     1.1305, 4.0461, 1.0343,
                     1.123, 4.0483, 1.0374,
                     1.1117, 4.0182, 1.0376,
                     1.1152, 4.048, 1.0431,
                     1.1215, 4.025, 1.0486,
                     1.1002, 4.012, 1.0491,
                     1.0927, 4.0015, 1.0464,
                     1.0822, 4.0037, 1.0367,
                     1.0824, 4.0131, 1.0338,
                     1.0772, 4.0185, 1.0383,
                     1.0743, 4.0022, 1.0323,
                     1.07, 3.9683, 1.0252,
                     1.0723, 3.9891, 1.0283,
                     1.0814, 4.0187, 1.0297,
                     1.0711, 4.031, 1.0327,
                     1.0579, 4.0178, 1.0321,
                     1.0564, 4.0105, 1.0344,
                     1.0552, 4.0136, 1.0373,
                     1.057, 4.0181, 1.039,
                     1.0774, 4.0167, 1.0447,
                     1.0862, 4.0171, 1.0438,
                     1.0847, 4.0578, 1.0438,
                     1.083, 4.0647, 1.04,
                     1.0755, 4.0608, 1.0426,
                     1.0759, 4.0854, 1.0463,
                     1.0845, 4.0938, 1.0439,
                     1.0856, 4.0978, 1.0476,
                     1.0973, 4.0763, 1.0498,
                     1.0985, 4.0858, 1.0517,
                     1.095, 4.1131, 1.0491,
                     1.0912, 4.1215, 1.0561,
                     1.0776, 4.1272, 1.0552,
                     1.0677, 4.1288, 1.0595,
                     1.0592, 4.1466, 1.0627,
                     1.0635, 4.1392, 1.066,
                     1.0557, 4.1281, 1.0615,
                     1.0572, 4.1539, 1.0648,
                     1.0613, 4.1371, 1.0636,
                     1.0578, 4.1365, 1.0655,
                     1.0738, 4.1368, 1.0703,
                     1.086, 4.1181, 1.0691,
                     1.0963, 4.1178, 1.07,
                     1.1069, 4.1397, 1.0697,
                     1.1124, 4.1727, 1.0694,
                     1.1168, 4.1582, 1.0736,
                     1.1227, 4.1563, 1.0724,
                     1.124, 4.1524, 1.0636,
                     1.1317, 4.1535, 1.0745,
                     1.1346, 4.1615, 1.0773,
                     1.1328, 4.1662, 1.0755,
                     1.1298, 4.1746, 1.0727,
                     1.1298, 4.1812, 1.0718,
                     1.1387, 4.1695, 1.0796,
                     1.1372, 4.1909, 1.0707,
                     1.1415, 4.1872, 1.0631,
                     1.1408, 4.1832, 1.0626,
                     1.1381, 4.1768, 1.0576,
                     1.1328, 4.1916, 1.0559,
                     1.1314, 4.218, 1.0465,
                     1.1297, 4.2008, 1.0457,
                     1.1275, 4.1822, 1.0428]


class ReturnsArrayTest(TestCase):

    def test_returns_array_returns_2x2_ndarray_with_diff_log_values(self):
        """
        """
        mock_asset_price_list = [1.0, 2.0, 1.5, 2.7]

        result = returns_array(2, mock_asset_price_list)

        self.assertEqual(type(result), np.ndarray)
        self.assertEqual(np.allclose(result, np.array([[0.405465, 0.300105]])), True)

    def test_returns_array_returns_10x3_ndarray_with_diff_log_values(self):
        """
        """
        mock_asset_price_list = [0.90533764, 0.04445307, 0.99678426, 0.17169811, 0.10199513,
                                 0.56025666, 0.04222177, 0.40399827, 0.8063998, 0.48682562,
                                 0.32202157, 0.26875212, 0.85752802, 0.38767992, 0.03249421,
                                 0.81413776, 0.2491216 , 0.72369488, 0.8305651, 0.92541793,
                                 0.24879266, 0.60131099, 0.43558035, 0.4581097, 0.97153666,
                                 0.89719849, 0.07971699, 0.85246873, 0.5229861, 0.19561174]

        result = returns_array(3, mock_asset_price_list)

        self.assertEqual(type(result), np.ndarray)
        self.assertEqual(np.allclose(result, np.array([[-1.66257, 0.830491, -0.576139],
                                                       [-1.4028, 1.37649, 0.364185],
                                                       [2.44497, -0.226792, -1.09879],
                                                       [0.566148, 0.185562, -2.11273],
                                                       [-0.0519243, -0.442239, 3.10331],
                                                       [0.0199767, 1.3123, -1.06775],
                                                       [-0.322994, -0.753566, 0.610489],
                                                       [0.479767, 0.722598, -1.74863],
                                                       [-0.130742, -0.539722, 0.897649]])), True)

    def test_returns_array_from_less_than_2_assets_raises_value_error(self):
        """
        """
        mock_asset_price_list = [1.0, 2.0, 1.5]

        with self.assertRaises(ValueError):
            _ = returns_array(1, mock_asset_price_list)

    def test_returns_array_from_price_number_not_divisible_by_no_of_assets_raises_value_error(self):
        """
        """
        mock_asset_price_list = [1.0, 2.0, 1.5]

        with self.assertRaises(ValueError):
            _ = returns_array(2, mock_asset_price_list)

    def test_returns_array_from_price_number_that_is_too_low_raises_value_error(self):
        """
        """
        mock_asset_price_list = [1.0, 2.0, 1.5]

        with self.assertRaises(ValueError):
            _ = returns_array(3, mock_asset_price_list)

    def test_returns_array_from_nonpositive_prices_raises_value_error(self):
        """
        """
        mock_asset_price_list = [1.0, 2.0, 1.5, 0.0]

        with self.assertRaises(ValueError):
            _ = returns_array(2, mock_asset_price_list)


class CholeskyDecompositionOfCovarianceMatrixTest(TestCase):

    def test_cholesky_decomposition_of_covariance_matrix_on_returns_array(self):
        """
        """
        mock_ret_array = MOCK_RETURN_ARRAY
        lower_triangular, upper_triangular =\
            cholesky_decomposition_of_covariance_matrix(mock_ret_array)

        with self.subTest('First matrix is lower triangular'):
            self.assertEqual(np.allclose(lower_triangular, np.tril(lower_triangular)), True)

        with self.subTest('Second matrix is upper triangular'):
            self.assertEqual(np.allclose(upper_triangular, np.triu(upper_triangular)), True)

        with self.subTest('Matrix product is equal to returns covariance matrix'):
            self.assertEqual(np.allclose(np.matrix(lower_triangular) * np.matrix(upper_triangular),
                                         np.cov(mock_ret_array, rowvar=False)), True)

    def test_cholesky_decomp_of_covariance_matrix_with_const_returns_raises_value_error(self):
        """
        """
        mock_ret_array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        with self.assertRaises(ValueError):
            _ = cholesky_decomposition_of_covariance_matrix(mock_ret_array)


class LDLDecompositionOfCovarianceMatrixTest(TestCase):

    def test_ldl_decomposition_of_covariance_matrix_on_returns_array(self):
        """
        """
        mock_ret_array = MOCK_RETURN_ARRAY
        lower_unit_triangular, diagonal, upper_unit_triangular =\
            ldl_decomposition_of_covariance_matrix(mock_ret_array)
        vector_length = diagonal.shape[0]

        with self.subTest('First matrix is lower triangular'):
            self.assertEqual(np.allclose(lower_unit_triangular, np.tril(lower_unit_triangular)),
                             True)

        with self.subTest('First matrix diagonal is all ones'):
            self.assertEqual(np.allclose(vector_length * [1.0], np.diag(lower_unit_triangular)),
                             True)

        with self.subTest('Second matrix is diagonal'):
            self.assertEqual(np.allclose(diagonal, np.diag(np.diag(diagonal))), True)

        with self.subTest('Third matrix is upper triangular'):
            self.assertEqual(np.allclose(upper_unit_triangular, np.triu(upper_unit_triangular)),
                             True)

        with self.subTest('Third matrix diagonal is all ones'):
            self.assertEqual(np.allclose(vector_length * [1.0], np.diag(upper_unit_triangular)),
                             True)

        with self.subTest('Matrix product is equal to returns covariance matrix'):
            self.assertEqual(np.allclose(np.matrix(lower_unit_triangular) * np.matrix(diagonal) *
                                         np.matrix(upper_unit_triangular),
                                         np.cov(mock_ret_array, rowvar=False)), True)

    def test_ldl_decomposition_of_covariance_matrix_with_const_returns_raises_value_error(self):
        """
        """
        mock_ret_array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        with self.assertRaises(ValueError):
            _ = ldl_decomposition_of_covariance_matrix(mock_ret_array)


class CorrelatedRandomVectorsUsingCholeskyDecompositionTest(TestCase):

    def test_correlated_random_vectors_using_cholesky_on_returns_array_with_given_returns(self):
        """
        """
        mock_number_of_vectors = 100000
        mock_ret_array = MOCK_RETURN_ARRAY
        result = correlated_random_vectors_using_cholesky_decomposition(mock_ret_array,
                                                                        mock_number_of_vectors)

        #  Since result is composed of random numbers, we check if the distribution is as expected
        #  by comparing covariance matrices of result and original data.
        result_cov = np.cov(result, rowvar=False)
        data_cov = np.cov(mock_ret_array, rowvar=False)

        self.assertEqual(np.allclose(result_cov, data_cov, rtol=1e-01), True)

    def test_correlated_random_vectors_using_cholesky_on_const_returns_raises_value_error(self):
        """
        """
        mock_number_of_vectors = 10
        mock_ret_array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        with self.assertRaises(ValueError):
            _ = correlated_random_vectors_using_cholesky_decomposition(mock_ret_array,
                                                                       mock_number_of_vectors)


class CorrelatedRandomVectorsTest(TestCase):

    def test_correlated_random_vectors_on_return_array_with_constant_returns(self):
        """
        """
        mock_number_of_vectors = 10
        mock_ret_array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        result = correlated_random_vectors(mock_ret_array, mock_number_of_vectors)

        self.assertEqual(np.allclose(result, np.array(10 * [[0.0, 0.0, 0.0]])), True)

    def test_correlated_random_vectors_on_return_array_with_given_returns(self):
        """
        """
        mock_number_of_vectors = 100000
        mock_ret_array = MOCK_RETURN_ARRAY
        result = correlated_random_vectors(mock_ret_array, mock_number_of_vectors)

        #  Since result is composed of random numbers, we check if the distribution is as expected
        #  by comparing covariance matrices of result and original data.
        result_cov = np.cov(result, rowvar=False)
        data_cov = np.cov(mock_ret_array, rowvar=False)

        self.assertEqual(np.allclose(result_cov, data_cov, rtol=1e-01), True)


class CalculateTest(TestCase):

    def __init__(self, *args, **kwargs):
        super(CalculateTest, self).__init__(*args, **kwargs)

        self.Params = namedtuple('Params', ['number_of_assets', 'alpha', 'time_perspective'])
        self.Data = namedtuple('Data', ['asset_prices', 'asset_weights'])

    def test_calculate_on_example_data_compared_with_parametric_var_computation(self):
        """
        """
        mock_parameters = self.Params(number_of_assets=3,
                                      alpha=99.0,
                                      time_perspective=20)
        mock_data = self.Data(
            asset_prices=MOCK_ASSET_PRICES,
            asset_weights=[0.2, 0.3, 0.5])

        result = calculate(mock_parameters, mock_data)["result"]

        weights_array = np.matrix(mock_data.asset_weights)
        parametric_volatility =\
            (weights_array * np.cov(returns_array(mock_parameters.number_of_assets,
                                                  mock_data.asset_prices),
                                    rowvar=False) * weights_array.T)[0, 0]
        z_score = stats.norm.ppf(mock_parameters.alpha / 100.0)
        parametric_result = sqrt(parametric_volatility) * z_score * sum(mock_data.asset_weights) *\
            mock_parameters.time_perspective

        self.assertAlmostEqual(result, parametric_result, places=2)

    def test_calculate_from_less_than_2_assets_raises_value_error(self):
        """
        """
        mock_parameters = self.Params(number_of_assets=1,
                                      alpha=99.0,
                                      time_perspective=20)
        mock_data = self.Data(
            asset_prices=[1.0, 1.2, 1.1, 1.1, 0.9, 1.3],
            asset_weights=[1.0])

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)

    def test_calculate_with_invalid_alpha_parameter_raises_value_error(self):
        """
        """
        mock_parameters = self.Params(number_of_assets=3,
                                      alpha=100.1,
                                      time_perspective=20)
        mock_data = self.Data(
            asset_prices=[1.0, 2.0, 3.0, 1.1, 1.8, 3.2, 1.5, 1.7, 3.1, 1.1, 2.0, 3.4,
                          1.2, 2.3, 3.8, 1.0, 2.5, 4.0, 0.9, 2.3, 4.1, 0.8, 2.2, 3.9],
            asset_weights=[0.2, 0.3, 0.5])

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)

    def test_calculate_with_invalid_time_perspective_raises_value_error(self):
        """
        """
        mock_parameters = self.Params(number_of_assets=3,
                                      alpha=99.0,
                                      time_perspective=0)
        mock_data = self.Data(
            asset_prices=[1.0, 2.0, 3.0, 1.1, 1.8, 3.2, 1.5, 1.7, 3.1, 1.1, 2.0, 3.4,
                          1.2, 2.3, 3.8, 1.0, 2.5, 4.0, 0.9, 2.3, 4.1, 0.8, 2.2, 3.9],
            asset_weights=[0.2, 0.3, 0.5])

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)

    def test_calculate_with_invalid_number_of_asset_weights_raises_value_error(self):
        """
        """
        mock_parameters = self.Params(number_of_assets=3,
                                      alpha=99.0,
                                      time_perspective=1)
        mock_data = self.Data(
            asset_prices=[1.0, 2.0, 3.0],
            asset_weights=[0.1, 0.3, 0.4, 0.2])

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)

    def test_calculate_with_negative_asset_weights_raises_value_error_____(self):
        """
        """
        mock_parameters = self.Params(number_of_assets=1,
                                      alpha=99.0,
                                      time_perspective=1)
        mock_data = self.Data(
            asset_prices=[1.0, 2.0, 3.0, 1.1, 1.8, 3.2, 1.5, 1.7, 3.1, 1.1, 2.0, 3.4,
                          1.2, 2.3, 3.8, 1.0, 2.5, 4.0, 0.9, 2.3, 4.1, 0.8, 2.2, 3.9],
            asset_weights=[-0.2, 0.3, 0.5])

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)


if __name__ == "__main__":
    main()
