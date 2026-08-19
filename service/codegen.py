import glob
from grpc.tools import protoc
import os

proto_files = []
for dirname, _, filenames in os.walk('.'):
    proto_files.extend(
        os.path.join(dirname, filename)
        for filename in filenames
        if filename.endswith('.proto') and not filename.startswith('.')
    )

protoc.main([
    '',
    '-I.',
    '--python_out=.',
    '--grpc_python_out=.',
] + proto_files)
