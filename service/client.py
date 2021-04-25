import grpc

import sys

sys.path.insert(0, 'service/')

from service_spec import fake_news_pb2
from service_spec import fake_news_pb2_grpc

channel = grpc.insecure_channel('localhost:7011')

stub = fake_news_pb2_grpc.fake_news_classificationStub(channel)

input_text = fake_news_pb2.InputMessage(value="This is sample text ")

response = stub.classify(input_text)

print(response.result)
