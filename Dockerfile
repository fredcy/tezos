FROM alpine:3.6

LABEL distro_style="apk" distro="alpine" distro_long="alpine-3.6" arch="x86_64" operatingsystem="linux"

RUN adduser -S tezos && \
    apk update && \
    apk upgrade && \
    apk add sudo bash libssl1.0 libsodium libev gmp git snappy && \
    apk add leveldb \
      --update-cache \
      --repository http://nl.alpinelinux.org/alpine/edge/testing && \
    rm -f /var/cache/apk/* && \
    echo 'tezos ALL=(ALL:ALL) NOPASSWD:ALL' > /etc/sudoers.d/tezos && \
    chmod 440 /etc/sudoers.d/tezos && \
    chown root:root /etc/sudoers.d/tezos && \
    sed -i 's/^Defaults.*requiretty//g' /etc/sudoers
USER tezos

COPY . /home/tezos
WORKDIR /home/tezos

RUN sudo chown root:root bin/* && \
    sudo chmod a+rx bin/* && \
    sudo mv bin/* /usr/local/bin && \
    rmdir bin

RUN sudo cp scripts/docker_entrypoint.sh /usr/local/bin/tezos && \
    sudo cp scripts/docker_entrypoint.inc.sh \
            scripts/client_lib.inc.sh \
            /usr/local/bin/ && \
    sudo chmod a+rx /usr/local/bin/tezos

RUN sudo mkdir -p /var/run/tezos && \
    sudo chown tezos /var/run/tezos

ENV EDITOR=vi

VOLUME /var/run/tezos

ENTRYPOINT [ "/usr/local/bin/tezos" ]
