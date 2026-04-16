FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

RUN echo "nameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 1.1.1.1" > /etc/resolv.conf

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /opt/ml/models

EXPOSE 8000

CMD ["/bin/sh", "-c", "echo 'nameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 1.1.1.1' > /etc/resolv.conf && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
