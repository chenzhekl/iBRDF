FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN mkdir /root/workspace

RUN apt-get update && apt-get install -y openssh-server cmake git clang-format gdb tmux htop libopenexr-dev
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN echo "export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}" >> /etc/profile
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

