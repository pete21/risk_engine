from collections import namedtuple
from unittest import TestCase, main

from models.portfolio_optimization.model import create_portfolio_variance,\
    create_egain_to_sd_ratio, calculate, api_pb2

DECIMAL_PLACES = 2  # number of decimal places to check in AlmostEqual tests


class CreatePortfolioVarianceTest(TestCase):

    def test_create_portfolio_variance_from_identity_returns_expected_function(self):
        """Supplying variance-covariance values resulting in identity matrix should produce function
        that returns sum of squares.
        """
        mock_variance_covariance_list = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        result = create_portfolio_variance(3, mock_variance_covariance_list)

        self.assertTrue(callable(result))

        with self.assertRaises(ValueError):
            result([1, 2])

        self.assertEqual(result([1, 2, 3]), 1**2 + 2**2 + 3**2)
        self.assertEqual(result([-71, 12, 0]), (-71)**2 + 12**2 + 0**2)

    def test_create_portfolio_variance_returns_expected_function(self):
        """Supplying given variance-covariance values should produce function that when called with
        weight list returns return variance.
        """
        mock_variance_covariance_list = [14, 3, 8, 3, 21, 7, 8, 7, 8]

        result = create_portfolio_variance(3, mock_variance_covariance_list)

        self.assertTrue(callable(result))

        with self.assertRaises(ValueError):
            result([1, 2, 3, 4])

        self.assertAlmostEqual(result([0.2, 0.3, 0.5]), 8.51, places=DECIMAL_PLACES)
        self.assertAlmostEqual(result([0.7, 0.1, 0.2]), 10.33, places=DECIMAL_PLACES)

    def test_create_variance_from_less_than_2_assets_raises_value_error(self):
        """Calling with number of assets that is less than 2 should raise ValueError exception.
        """
        mock_variance_covariance_list = [1.0]

        with self.assertRaises(ValueError):
            _ = create_portfolio_variance(1, mock_variance_covariance_list)

    def test_create_variance_from_invalid_number_of_values_raises_value_error(self):
        """Calling with number of variance-covariance values not equal to squared number of
        assets should raise ValueError exception.
        """
        mock_variance_covariance_list = [1.0, 2.0, 3.0]

        with self.assertRaises(ValueError):
            _ = create_portfolio_variance(2, mock_variance_covariance_list)

    def test_create_variance_from_values_giving_not_symmetric_matrix_raises_value_error(self):
        """Calling with such a list of values that resulting covariance matrix is not symmetric
        should raise ValueError exception.
        """
        mock_variance_covariance_list = [1.0, 0.0, 0.01, 4.0]

        with self.assertRaises(ValueError):
            _ = create_portfolio_variance(2, mock_variance_covariance_list)

    def test_create_variance_from_values_giving_not_positive_semidefinite_matrix_raises_error(self):
        """Calling with such a list of values that resulting covariance matrix is not positive
        semidefinite should raise ValueError exception.
        """
        mock_variance_covariance_list = [1.0, 2.0, 1.0, 2.0]

        with self.assertRaises(ValueError):
            _ = create_portfolio_variance(2, mock_variance_covariance_list)


class CreateEgainToSdRatioTest(TestCase):

    def test_create_egain_to_sd_ratio_for_2_given_assets_returns_expected_function(self):
        """Supplying given variance-covariance values should produce function that when called with
        weight list returns negated expected gain to SD ratio.
        """
        mock_variance_covariance_list = [0.5, -1, -1, 2]
        mock_returns_list = [0.08, 0.06]

        result = create_egain_to_sd_ratio(2, mock_variance_covariance_list, mock_returns_list)

        self.assertTrue(callable(result))

        with self.assertRaises(ValueError):
            result([1, 2, 3])

        self.assertAlmostEqual(result([0.6, 0.4]), -0.509117, places=DECIMAL_PLACES)
        self.assertAlmostEqual(result([0.42, 0.58]), -0.130719, places=DECIMAL_PLACES)

    def test_create_egain_to_sd_ratio_for_3_given_assets_returns_expected_function(self):
        """Supplying given variance-covariance values should produce function that when called with
        weight list returns negated expected gain to SD ratio.
        """
        mock_variance_covariance_list = [14, 3, 8, 3, 21, 7, 8, 7, 8]
        mock_returns_list = [0.08, 0.06, 0.17]

        result = create_egain_to_sd_ratio(3, mock_variance_covariance_list, mock_returns_list)

        self.assertTrue(callable(result))

        with self.assertRaises(ValueError):
            result([1.0, 2.5])

        self.assertAlmostEqual(result([0.6, 0.25, 0.15]), -0.0288694)
        self.assertAlmostEqual(result([0.1, 0.35, 0.55]), -0.0410218)

    def test_create_ratio_from_number_of_values_unequal_to_that_of_assets_raises_value_error(self):
        """Calling with number of values not equal to the number of assets should raise ValueError
        exception.
        """
        mock_variance_covariance_list = [1.0, 0.0, 0.0, 1.0]
        mock_returns_list = [8.5, 12.0, 3.33]

        with self.assertRaises(ValueError):
            create_egain_to_sd_ratio(1, mock_variance_covariance_list, mock_returns_list)


class CalculateTest(TestCase):

    def __init__(self, *args, **kwargs):
        super(CalculateTest, self).__init__(*args, **kwargs)

        self.Params = namedtuple('Params', ['number_of_assets', 'type', 'algorithm'])
        self.Data = namedtuple('Data', ['variances', 'returns', 'additional_constraints'])
        self.Constraint = namedtuple('Constraint', ['weight_index', 'interval_lower_bound',
                                                    'interval_upper_bound'])

    def test_calculate_optimal_portfolio_2_wrt_variance(self):
        """Calculating minimal variance for given 4 variance-covariance values using all 3
        available algorithms should return provided results with specified precision in each of 3
        cases.
        """
        mock_data = self.Data(
            variances=[14.0, 9.0, 9.0, 21.0],
            returns=[],
            additional_constraints=[])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=2,
                                              type=api_pb2.ModelParams.VARIANCE,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, 12.5294, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[0], 0.705882, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_3_wrt_variance(self):
        """Calculating minimal variance for given 9 variance-covariance values using all 3
        available algorithms should return provided results with specified precision in each of 3
        cases.
        """
        mock_data = self.Data(
            variances=[14.0, 3.0, 8.0, 3.0, 21.0, 7.0, 8.0, 7.0, 8.0],
            returns=[],
            additional_constraints=[])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=3,
                                              type=api_pb2.ModelParams.VARIANCE,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, 7.91892, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[0], 0.0540542, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[1], 0.0810812, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_2_wrt_variance_with_additional_constr(self):
        """Calculating minimal variance for given 4 variance-covariance values and additional
        constraints, using all 3 available algorithms, should return provided results with specified
        precision in each of 3 cases.
        """
        mock_data = self.Data(
            variances=[14.0, 9.0, 9.0, 21.0],
            returns=[],
            additional_constraints=
            [self.Constraint(weight_index=1, interval_lower_bound=0, interval_upper_bound=0.5)])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=2,
                                              type=api_pb2.ModelParams.VARIANCE,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, 13.25, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[0], 0.5, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_3_wrt_variance_with_additional_constr(self):
        """Calculating minimal variance for given 9 variance-covariance values and additional
        constraints, using all 3 available algorithms, should return provided results with specified
        precision in each of 3 cases.
        """
        mock_data = self.Data(
            variances=[14.0, 3.0, 8.0, 3.0, 21.0, 7.0, 8.0, 7.0, 8.0],
            returns=[],
            additional_constraints=
            [self.Constraint(weight_index=2, interval_lower_bound=0.1, interval_upper_bound=1),
             self.Constraint(weight_index=3, interval_lower_bound=0.1, interval_upper_bound=0.4)])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=3,
                                              type=api_pb2.ModelParams.VARIANCE,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, 8.47034, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[0], 0.358621, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[1], 0.241379, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_2_wrt_expected_return_to_SD_ratio(self):
        """Calculating minimal negated expected return to standard deviation ratio for given 4
        variance-covariance values using all 3 available algorithms should return provided results
        with specified precision in each of 3 cases.
        """
        mock_data = self.Data(
            variances=[14.0, 9.0, 9.0, 21.0],
            returns=[0.08, 0.06],
            additional_constraints=[])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=2,
                                              type=api_pb2.ModelParams.RETURN_TO_SD_RATIO,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, -0.0214935, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[0], 0.904654, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_3_wrt_expected_return_to_SD_ratio(self):
        """Calculating minimal negated expected return to standard deviation ratio for given 9
        variance-covariance values using all 3 available algorithms should return provided results
        with specified precision in each of 3 cases.
        """
        mock_data = self.Data(
            variances=[14.0, 3.0, 8.0, 3.0, 21.0, 7.0, 8.0, 7.0, 8.0],
            returns=[0.08, 0.06, 0.17],
            additional_constraints=[])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=3,
                                              type=api_pb2.ModelParams.RETURN_TO_SD_RATIO,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, -0.0601039, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[0], 2.85521e-6, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[1], 2.89959e-6, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_2_wrt_expected_return_to_SD_ratio_with_add_constr(self):
        """Calculating minimal negated expected return to standard deviation ratio for given 4
        variance-covariance values and additional constraints, using all 3 available algorithms,
        should return provided results with specified precision in each of 3 cases.
        """
        mock_data = self.Data(
            variances=[14.0, 9.0, 9.0, 21.0],
            returns=[0.08, 0.06],
            additional_constraints=
            [self.Constraint(weight_index=1, interval_lower_bound=0, interval_upper_bound=0.5)])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=2,
                                              type=api_pb2.ModelParams.RETURN_TO_SD_RATIO,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, -0.0192304, places=DECIMAL_PLACES)

                # TRUST_CONSTR method weight approximation is rather crude in this case
                if alg == api_pb2.ModelParams.TRUST_CONSTR:
                    self.assertAlmostEqual(result.x[0], 0.499991, places=1)
                else:
                    self.assertAlmostEqual(result.x[0], 0.499991, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_3_wrt_expected_return_to_SD_ratio_with_add_constr(self):
        """Calculating minimal negated expected return to standard deviation ratio for given 9
        variance-covariance values and additional constraints, using all 3 available algorithms,
        should return provided results with specified precision in each of 3 cases.
        """
        mock_data = self.Data(
            variances=[14.0, 3.0, 8.0, 3.0, 21.0, 7.0, 8.0, 7.0, 8.0],
            returns=[0.08, 0.06, 0.17],
            additional_constraints=
            [self.Constraint(weight_index=2, interval_lower_bound=0.1, interval_upper_bound=1),
             self.Constraint(weight_index=3, interval_lower_bound=0.1, interval_upper_bound=0.4)])

        for alg in [api_pb2.ModelParams.TRUST_CONSTR,
                    api_pb2.ModelParams.SLSQP,
                    api_pb2.ModelParams.COBYLA]:
            with self.subTest(algorithm=['TRUST_CONSTR', 'SLSQP', 'COBYLA'][alg]):
                mock_parameters = self.Params(number_of_assets=3,
                                              type=api_pb2.ModelParams.RETURN_TO_SD_RATIO,
                                              algorithm=alg)

                result = calculate(mock_parameters, mock_data)

                self.assertAlmostEqual(result.fun, -0.0383785, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[0], 0.411143, places=DECIMAL_PLACES)
                self.assertAlmostEqual(result.x[1], 0.188859, places=DECIMAL_PLACES)

    def test_calculate_optimal_portfolio_using_unknown_algorithm_raises_value_error(self):
        """Calling calculate with unknown optimization algorithm should raise ValueError exception.
        """
        mock_data = self.Data(
            variances=[14.0, 9.0, 9.0, 21.0],
            returns=[0.08, 0.06],
            additional_constraints=[])

        mock_parameters = self.Params(number_of_assets=3,
                                      type=api_pb2.ModelParams.RETURN_TO_SD_RATIO,
                                      algorithm=42)

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)

    def test_calculate_optimal_portfolio_using_unknown_minimization_type_raises_value_error(self):
        """Calling calculate with unknown minimization type should raise ValueError exception.
        """
        mock_data = self.Data(
            variances=[14.0, 9.0, 9.0, 21.0],
            returns=[0.08, 0.06],
            additional_constraints=[])

        mock_parameters = self.Params(number_of_assets=3,
                                      type=42,
                                      algorithm=api_pb2.ModelParams.TRUST_CONSTR)

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)

    def test_calculate_optimal_portfolio_with_invalid_constraint_raises_value_error(self):
        """Calling calculate with invalid constraint should raise ValueError exception.
        """
        mock_data = self.Data(
            variances=[14.0, 9.0, 9.0, 21.0],
            returns=[0.08, 0.06],
            additional_constraints=
            [self.Constraint(weight_index=1, interval_lower_bound=0.5, interval_upper_bound=0.45)])

        mock_parameters = self.Params(number_of_assets=2,
                                      type=api_pb2.ModelParams.VARIANCE,
                                      algorithm=api_pb2.ModelParams.TRUST_CONSTR)

        with self.assertRaises(ValueError):
            _ = calculate(mock_parameters, mock_data)


if __name__ == "__main__":
    main()
