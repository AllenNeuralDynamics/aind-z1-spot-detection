# hash:sha256:371bf441e56d9d9ffcfd52a9973e12488c057e385eb66336659d37be6fbbac67
FROM registry.codeocean.allenneuraldynamics.org/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ARG new_pred_version="1.0.0"

RUN pip install --no-cache-dir \
    jupyterlab \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git \
    numpy==1.26.4 \
    scikit-image==0.22.0 \
    matplotlib==3.8.2 \
    scipy==1.10.1 \
    cupy-cuda11x==13.0.0 \
    torch==2.2.2 \
    boto3 \
    xmltodict \
    dask[array] \
    dask-image \
    neuroglancer

ADD "https://github.com/coder/code-server/releases/download/v4.95.3/code-server-4.95.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.95.3-linux-amd64/bin/code-server  /usr/bin/code-server

RUN mkdir -p /.vscode/extensions \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension REditorSupport.R \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension continue.continue \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reageyao.bioSyntax \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev
