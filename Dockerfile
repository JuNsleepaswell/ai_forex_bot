# Start from the official NVIDIA image that supports your GPU
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# As the root user, run the pip install commands
# This bakes the libraries directly into our new image
RUN pip install --no-cache-dir \
    matplotlib \
    keras --upgrade

# Set the default working directory when the container starts
WORKDIR /workspace