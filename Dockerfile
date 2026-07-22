FROM python:3.10

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install jupyterlab notebook

COPY . /workspace

RUN pip install .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token="]