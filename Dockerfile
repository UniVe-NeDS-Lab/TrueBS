#FROM nvidia/cuda:12.8.0-devel-ubuntu20.04
FROM nvcr.io/nvidia/ai-workbench/python-cuda117:1.0.6

WORKDIR truebs/

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN echo "#!/bin/bash" > entrypoint.sh
#RUN echo "python3 TrueBS.py"
RUN echo "while true; do sleep 100; done" >> entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/truebs/entrypoint.sh"]