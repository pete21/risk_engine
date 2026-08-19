all: models/iris/iris_model.pickle
	python3 codegen.py

models/iris/iris_model.pickle: models/iris/train.py
	python3 $<

clean:
	find \( -name '*_pb2.py' -o -name '*_pb2_grpc.py' -o -name '__pycache__' -o -name 'iris_model.pickle' \) -exec rm -r {} +

.PHONY: all clean
