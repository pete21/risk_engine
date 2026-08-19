import grpc

import risk_pb2
import risk_pb2_grpc


class GrpcClient():
    def __init__(self, grpc_host, grpc_port):
        """
        """

        channel = grpc.insecure_channel('%s:%d' % (grpc_host, grpc_port))
        self.stub = risk_pb2_grpc.RiskEngineStub(channel)

    def send_request(self, model_name, model_parameters=None, model_input=None, msg_prefix=""):
        req = risk_pb2.CalculationRequest(model_name=model_name, model_parameters=model_parameters,
                                          model_input=model_input)

        print(msg_prefix+'Sending {} request.'.format(model_name))

        try:
            res = self.stub.Calculate(req)
        except Exception as err:
            raise GrpcClientException(err)

        for key, value in sorted(res.values.items()):
            print(msg_prefix+'{}: {}'.format(key, value))
        print()

        return res.values


class GrpcClientException(Exception):
    pass
