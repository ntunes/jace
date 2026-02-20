FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libxslt-dev libxml2-dev libffi-dev libssl-dev \
        iputils-ping traceroute dnsutils curl netcat-openbsd \
        iproute2 net-tools openssh-client mtr-tiny \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

RUN useradd --system --create-home jace \
    && mkdir -p /data /config \
    && chown jace:jace /data /config
USER jace

EXPOSE 8080

ENTRYPOINT ["jace"]
