FROM continuumio/miniconda3
# use bash as default shell (personal preference)
SHELL ["/bin/bash", "-ec"]
# general system updates, install git
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
# default port for streamlit = 8501
EXPOSE 8501
# must run as not-root user (CERN OpenShift requirement), set that up
ENV USER=appuser
RUN useradd -p $(openssl passwd -1 password) --create-home ${USER}
RUN adduser ${USER} sudo
RUN chown -R ${USER} /home/${USER}
RUN chown -R ${USER} /opt/conda
USER ${USER}
# copy our code into the container
ENV WORKDIR=/home/${USER}/app
WORKDIR ${WORKDIR}
COPY . ${WORKDIR}
# set up conda: update, create env, set commands to run in env
RUN conda update -y -n base -c defaults conda
RUN conda create -y --name pytta -c conda-forge --file environment.yml
RUN conda init bash
SHELL ["conda", "run", "-n", "pytta", "/bin/bash", "-c"]
# launch the site when we start the container
ENTRYPOINT ["/opt/conda/envs/pytta/bin/streamlit", "run", "website.py", "--server.port=8501"]
