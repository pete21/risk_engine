import argparse
import collections
import concurrent.futures
import importlib
import os
import random
import re
import sys
import threading
import time
import traceback

import grpc

import risk_pb2
import risk_pb2_grpc


class ModelLoader(object):
    _VALID_NAME_RE = re.compile('^[a-z_]*[a-z0-9_]*$')

    def __init__(self):
        self._lock = threading.Lock()
        self._models = self._Dict()

    def get(self, model_name):
        if not self._VALID_NAME_RE.search(model_name):
            raise ValueError('Invalid model name: ' + model_name)
        with self._lock:
            loader = self._models[model_name]
        return loader.get()

    class _Dict(collections.defaultdict):
        def __missing__(self, model_name):
            return ModelLoader._LoaderThunk(model_name)

    class _LoaderThunk(object):
        def __init__(self, model_name):
            self._lock = threading.Lock()
            self._get = lambda: self._doGet(model_name)

        def get(self):
            with self._lock:
                return self._get()

        def _doGet(self, model_name):
            try:
                module = importlib.import_module('models.%s.model' % model_name)
                self._get = lambda: module
            except ModuleNotFoundError as ex:
                self._get = self._raisingLambda(ValueError(
                    'Unknown model: {} ({})'.format(model_name, ex)))
            except Exception as ex:
                self._get = self._raisingLambda(ex)
            return self._get()

        @staticmethod
        def _raisingLambda(ex):
            def execute():
                raise ex
            return execute


class RiskEngine(risk_pb2_grpc.RiskEngineServicer):
    def __init__(self, *args, **kw):
        super(*args, **kw)
        self._models = ModelLoader()
        self._next_call_id = 0
        self._write_lock = threading.Lock()

    def _write(self, call_id, fmt, *args, **kw):
        message = fmt.format(*args, **kw)
        with self._write_lock:
            if call_id is None:
                call_id = self._next_call_id
                self._next_call_id += 1
            sys.stderr.write('[{:3}] '.format(call_id))
            sys.stderr.write(message)
            sys.stderr.write('\n')
            sys.stderr.flush()
        return call_id

    def Calculate(self, req, ctx):
        call_id = None
        try:
            delay = random.random() * 5.0
            call_id = self._write(
                call_id, 'Calculate({}) (simulated latency: {} s)',
                req.model_name, delay)
            time.sleep(delay)
            model_name = req.model_name
            model = self._models.get(model_name)
            res = model.calculate(self._decodeData(req, model, 'parameters'),
                                  self._decodeData(req, model, 'input'))
            sys.stderr.write('[{:3}] {}: {}\n'.format(
                    call_id, req.model_name, res))
            return risk_pb2.CalculationResponse(values=res)
        except:
            self._write(call_id, traceback.format_exc())
            raise
        finally:
            sys.stderr.flush()

    @staticmethod
    def _decodeData(req, model, name):
        decoder = getattr(model, 'decode_' + name, None)
        value = getattr(req, 'model_' + name)
        if decoder is not None:
            return decoder('model_' + name, value)
        elif value:
            raise TypeError('Unexpected model_{} request field'.format(name))
        else:
            return value


def serve(port, max_workers):
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers))
    risk_pb2_grpc.add_RiskEngineServicer_to_server(RiskEngine(), server)
    server.add_insecure_port('[::]:{port}'.format(port=port))
    server.start()
    try:
        while True:
            time.sleep(24 * 60 * 60)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--port', type=int, help='port number',
                        required=False, default=8080)
    parser.add_argument('--max_workers', type=int, help='# max workers',
                        required=False, default=10)
    args = parser.parse_args()

    serve(port=args.port, max_workers=args.max_workers)
