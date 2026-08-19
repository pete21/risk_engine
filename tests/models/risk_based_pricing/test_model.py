# -*- coding: utf-8 -*-

from collections import namedtuple
from unittest import TestCase, main

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from models.logistic_regression.model import load_model_components, calculate,\
    LR_MODEL_PATH, WOE_BINS_PATH


class LoadModelComponentsTest(TestCase):

    def test_load_model_components_returns_expected_data_structures(self):

        result = load_model_components(LR_MODEL_PATH, WOE_BINS_PATH)

        with self.subTest('Result is a tuple of length 3.'):
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)

        lr_model, woe_bins, model_attribute_list = result

        with self.subTest('First component is a LogisticRegression object with callable field'
                          'predict_proba.'):
            self.assertIsInstance(lr_model, LogisticRegression)
            self.assertTrue(callable(lr_model.predict_proba))

        with self.subTest('Second component is a dictionary containing data frames.'):
            self.assertIsInstance(woe_bins, dict)
            for woe_value in woe_bins.values():
                self.assertIsInstance(woe_value, DataFrame)

        with self.subTest('Third component is a list of keys of this dictionary.'):
            self.assertIsInstance(model_attribute_list, list)
            self.assertCountEqual(model_attribute_list, [*woe_bins])

    def test_load_model_components_from_invalid_model_path_raises_value_error(self):

        with self.assertRaises(ValueError):
            _ = load_model_components('/this/path/is/not/valid', WOE_BINS_PATH)

    def test_load_model_components_from_invalid_woe_path_raises_value_error(self):

        with self.assertRaises(ValueError):
            _ = load_model_components(LR_MODEL_PATH, '/this/path/is/not/valid')

    def test_load_model_components_from_invalid_woe_file_raises_value_error(self):

        with self.assertRaises(ValueError):
            _ = load_model_components(LR_MODEL_PATH, LR_MODEL_PATH)


class CalculateTest(TestCase):

    def __init__(self, *args, **kwargs):
        super(CalculateTest, self).__init__(*args, **kwargs)

        self.Data = namedtuple('Data', ['risk_attributes_list'])
        self.Dictionary = namedtuple('Dictionary', ['pairs'])
        self.Pair = namedtuple('Pair', ['key', 'value'])

    def convert_to_protobuf_style(self, list_of_dicts):
        return [self.Dictionary(pairs=[self.Pair(key=k, value=v) for k, v in d.items()])
                for d in list_of_dicts]

    def test_calculate_returns_expected_data_structures(self):
        mock_data = self.Data(
            risk_attributes_list=self.convert_to_protobuf_style(MOCK_RISK_ATTRIBUTES_LIST))

        result = calculate(None, mock_data)

        with self.subTest('Result is a list.'):
            self.assertIsInstance(result, list)

        with self.subTest('Result length is equal to risk attributes list length.'):
            self.assertEqual(len(result), len(MOCK_RISK_ATTRIBUTES_LIST))

        with self.subTest('All elements of result list are floats between 0 and 1.'):
            self.assertTrue(all(isinstance(item, float) and 0 <= item <= 1 for item in result))

    def test_calculate_from_empty_risk_attributes_list_raises_value_error(self):

        with self.assertRaises(ValueError):
            _ = calculate(None, self.Data(risk_attributes_list=[]))

    def test_calculate_from_invalid_risk_attributes_list_raises_value_error(self):

        with self.assertRaises(ValueError):
            _ = calculate(None, self.Data(
                risk_attributes_list=self.convert_to_protobuf_style(
                    MOCK_INVALID_RISK_ATTRIBUTES_LIST)))


if __name__ == "__main__":
    main()
