# Deploy models for Sirius project

To build Docker image you need `CA.pem` file which you may get with the following manual https://cloud.yandex.ru/docs/managed-mongodb/operations/connect/#get-ssl-cert

To run model use:
```
docker run -it --rm --name modelexec --env-file dockerdeploy/models/docker.env deploymodel:latest python3 /app/<MODEL_NAME>.py 62cac1e8dc6b0630c969e513 <FILE_NAME>.xlsx <NUM_CLUSTERS>
```
where:
- `<MODEL_NAME>.py` = name of the Python script with the model
- `<FILE_NAME>.xlsx` = Excel file with the data (two columns `id` and `text`)
- `<NUM_CLUSTERS>` = number of the clusters, required only for LDA and Deep K-Means models

Example of `docker.env` file:
```
S3_BUCKET=<name-of-S3-bucket>
SERVICE_NAME=s3
KEY=<key-for-access>
SECRET=<secret-for-access>
ENDPOINT=http://storage.yandexcloud.net

MONGODB_HOST=<mongo-host>:<mongo-port>
MONGODB_DATABASE=<db-name>
MONGODB_USERNAME=<mongo-user>
MONGODB_PASSWORD=<mongo-password>

```
