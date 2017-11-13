# ml-models

This is a companion project to [ml-apis](https://github.com/danielpoonwj/ml-api), a proof-of-concept workflow to expose predictive machine learning models as a HTTP API in **Python 3.6**.

This repository is the core codebase in which the models are defined and trained. The rationale for separating this from the API codebase is to:
1) Allow independent scaling or resource allocation. Resources required for training the models and serving the API can be vastly different.
2) Ensure HTTP API code is almost entirely decoupled from model definition for more flexibility, less spaghetti. 

## Getting Started
Once the models are trained, there are two modes of operation for saving and loading the trained models.

### Local
Saving and loading data would be on the local filesystem. The trained model would be saved into `/tmp/ml_models/{model_name}/v{version_number}/timestamp.pkl` by default. 

Overriding the default parent path (`/tmp/ml_models/`) can be done through an [environment variable](#environment-variables).

### AWS S3
Saving and loading data would be on AWS S3. The trained model would be saved into `{s3_bucket}/{model_name}/v{version_number}/timestamp.pkl`.

This mode requires the relevant [environment variables](#environment-variables) to be set.


## Installing
### Docker
The simplest method to try this project out is through Docker.

The `--env-file` flag loads an optional file for environment variables which are required for operating in aws mode but not necessary for local mode. 

The `-v` flag would attach a volume to allow the saved model to be available on the host machine in local mode. This is not relevant for aws mode. 

```bash
cd path/to/project

docker build -t temp/ml-models:latest .

docker run \
    --rm \
    -t \
    -v /tmp/ml_models:/tmp/ml_models \
    --env-file ./.env \
    temp/ml-models:latest
``` 

### Local Python Virtual Environment
For development or running locally, a virtual environment is strongly recommended - I personally use [pyenv](https://github.com/pyenv/pyenv).

If a `.env` file is present, the environment variables would automatically be applied in this process. 

```bash
cd path/to/project

# create and activate Python 3.6 virtual environment

pip install -r requirements.txt

python training_script.py
```  

## Environment Variables 
| Mode | Environment Variable | Purpose | Accepted Values |
| --- | --- | --- | --- |
| - |  ML_MODELS_MODE | Toggle mode of operation. If unset, defaults to local| local, aws |
| local | ML_MODELS_DIR | Parent directory to read/write models | any |
| aws | AWS_BUCKET_NAME | S3 Bucket to read/write models | any |
| aws | AWS_ACCESS_KEY_ID | S3 Credentials to be used by [`boto3`](http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables) | any |
| aws | AWS_SECRET_ACCESS_KEY | S3 Credentials to be used by [`boto3`](http://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables) | any |
