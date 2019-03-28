FROM ubuntu:latest

RUN mkdir ./app
WORKDIR ./app

RUN apt-get update && apt-get install -y wget curl unzip unrar build-essential python3 python3-pip

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN rm ~/anaconda.sh
# Set path to conda
ENV PATH /opt/conda/bin:$PATH
# Update Anaconda
RUN conda update conda && conda update anaconda && conda update --all

# # Install Jupyter theme
# RUN pip install msgpack jupyterthemes
# RUN jt -t grade3
# # Install other Python packages
# RUN conda install pymssql
# RUN pip install SQLAlchemy \
#     missingno \
#     json_tricks \
#     bcolz \
#     gensim \
#     elasticsearch \
#     psycopg2-binary
