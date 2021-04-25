
import grpc
from concurrent import futures
import time

import sys
sys.path.insert(0, 'service/')

from service_spec import fake_news_pb2
from service_spec import fake_news_pb2_grpc

import json

import test

class fake_news_classificationServicer(fake_news_pb2_grpc.fake_news_classificationServicer):

    def classifygit s(self, request, context):
        response = fake_news_pb2.OutputMessage()
        response.result = test.predict(request.value)
        return response


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
fake_news_pb2_grpc.add_fake_news_classificationServicer_to_server(fake_news_classificationServicer(), server)

print('Starting server. Listening on port 7011.')
server.add_insecure_port('0.0.0.0:7011')
server.start()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
