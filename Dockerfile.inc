#FROM intel/intel-extension-for-pytorch:2.1.10-xpu
#USER root

# TODO _VERSION arguments for dnnl etc. - use the same version as previously installed runtime libs
RUN apt-get update && apt-get install -y ninja-build intel-oneapi-dpcpp-cpp-2024.0 intel-oneapi-dnnl-devel-2024.0 intel-oneapi-mkl-devel-2024.0 python3-dev

# conda post-installation taken from https://github.com/ContinuumIO/docker-images/tree/main/miniconda3 (3-clause BSD license, "(c) 2012 Continuum Analytics, Inc. / http://continuum.io", see the link for the full license)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH

COPY environment_intel.yml /environment_intel.yml
RUN conda env create -f environment_intel.yml && rm environment_intel.yml
RUN echo "conda activate stylegan3" >> ~/.bashrc
RUN echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc

RUN apt-get update && apt-get install -y intel-oneapi-dpcpp-ct-2024.0
#RUN apt-get update && apt-get install -y intel-oneapi-vtune
