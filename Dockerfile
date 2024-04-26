FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0
RUN python -m pip install jupyter
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt


RUN jupyter notebook --generate-config
RUN echo "c=get_config()" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.InlineBackend.rc = { }" >> /root/.jupyter/jupyter_notebook_config.py
RUN mkdir -p /workspace/datasets/
RUN mkdir -p /workspace/dvio/
WORKDIR /workspace/dvio/
RUN apt install -y tree build-essential
RUN export PYTHONPATH="/workspace/dvio:$PYTHONPATH" >> ~/.bashrc