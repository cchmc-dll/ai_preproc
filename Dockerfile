FROM jupyter/minimal-notebook:latest

USER root

COPY requirements.txt /home/$NB_USER/src/requirements.txt

RUN pip install --default-timeout=60 -r /home/$NB_USER/src/requirements.txt

COPY --chown=1000:100 src /home/$NB_USER/src

COPY --chown=1000:100 preproc.args /home/$NB_USER/preproc.args

COPY --chown=1000:100 *.ipynb /home/$NB_USER/

COPY --chown=1000:100 launcher.py /home/$NB_USER/

WORKDIR /home/$NB_USER/

USER $NB_USER
#EXPOSE 80
