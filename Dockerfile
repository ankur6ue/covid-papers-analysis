FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime as pytorch_1_5_faiss

RUN apt-get update
RUN apt-get install -y curl wget

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# install blas and openmp (dev versions) - needed for FAISS
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y libomp-dev
# RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install FAISS
# Install MKL (copied from FAISS dockerfile)
# Install necessary build tools
RUN apt-get install -y make swig
# git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
# gcc
RUN apt-get update && apt-get -y install gcc
RUN apt-get update && apt-get -y install g++

WORKDIR /opt
RUN git clone https://github.com/facebookresearch/faiss.git
WORKDIR /opt/faiss
RUN ./configure --without-cuda
RUN make -j $(nproc)
RUN make -C python
# RUN make test
RUN make install
RUN make -C python install

## done with FAISS ##
# do it here because pymagnitude takes a long time to install..
FROM pytorch_1_5_faiss
# RUN pip install pymagnitude
RUN mkdir /app
COPY requirements.txt /app
RUN apt-get install nano
RUN pip install -r /app/requirements.txt
COPY / /app/
WORKDIR /app/
#ENV FLASK_APP covid_papers_browser_app:application
ENTRYPOINT [ "python" ]
CMD ["covid_papers_browser_app.py"]