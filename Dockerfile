FROM tensorflow/serving
COPY models/20231120-094457 /models/flower
ENV MODEL_NAME flower