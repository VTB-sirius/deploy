FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive

RUN useradd -ms /bin/bash --uid 1000 jupyter\
 && apt update\
 && apt install -y python3.8-dev python3.8-distutils gnupg wget software-properties-common curl\
 && ln -s /usr/bin/python3.8 /usr/local/bin/python3\
 && curl https://bootstrap.pypa.io/get-pip.py | python3

ENV LD_LIBRARY_PATH /usr/local/cuda-11.2/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
RUN apt-get update &&\
 apt-get install -y -q xserver-xorg-core wget &&\
 wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 &&\
 apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub &&\
 add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" &&\
 apt-get update &&\
 apt-get -y install cuda libcudnn8 nvidia-cuda-toolkit &&\
 exit
RUN pip install tensorflow-gpu==2.6

# base layer read xls and s3
RUN pip install pandas pandas xlwt xlrd fsspec s3fs boto3 openpyxl pymongo 

# layers for LDA model
RUN pip install gensim==3.8.3 razdel nltk pymystem3 tqdm pip scikit-learn

RUN apt install -y unzip &&\
  apt install -y default-jdk

RUN mkdir /app && cd /app &&\
  wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip &&\
  unzip mallet-2.0.8.zip && rm mallet-2.0.8.zip

RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/app/nltk_data')" ]

RUN [ "python3", "-c", "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data')" ]

ENV NLTK_DATA /app/nltk_data

ADD . $NLTK_DATA

RUN wget http://download.cdn.yandex.net/mystem/mystem-3.0-linux3.1-64bit.tar.gz &&\
  tar -xvf mystem-3.0-linux3.1-64bit.tar.gz && mv mystem /bin &&\
  rm mystem-3.0-linux3.1-64bit.tar.gz

RUN pip install pyarrow==6.0.0 sentence_transformers seaborn umap-learn datasets

COPY CA.pem /app

COPY models/*.py /app/
