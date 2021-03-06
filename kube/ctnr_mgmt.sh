run_apache_proxy_server() {
  IMAGE_NAME=$1
  CONTAINER_NAME=$2
  CONTAINER_NAME_DEV=$3
  BASE_PATH=$4

  docker build -t $IMAGE_NAME:latest apache
  docker ps --filter "name=$CONTAINER_NAME" -aq | xargs docker stop | xargs docker rm
  docker ps --filter "name=$CONTAINER_NAME_DEV" -aq | xargs docker stop | xargs docker rm
  # add |xargs docker rm to also remove them
  sleep 1

  docker run -p 8000:8000 --name=$CONTAINER_NAME -d \
    -v $BASE_PATH/apache/sites/:/usr/local/apache2/conf/sites \
    -v $BASE_PATH/apache/sites/:/usr/local/apache2/covid-demo \
    -it $IMAGE_NAME:latest

  docker run -p 8082:8082 --name=$CONTAINER_NAME_DEV -d \
  -v $BASE_PATH/apache/dev/:/usr/local/apache2/conf/sites \
  -v $BASE_PATH/apache/dev/:/usr/local/apache2/covid-demo \
  -it $IMAGE_NAME:latest
}

run_application_server() {
  IMAGE_NAME=$1
  CONTAINER_NAME=$2
  BASE_PATH_=$3
  docker build -t $IMAGE_NAME:latest .
  docker ps --filter "name=$CONTAINER_NAME" -aq | xargs docker stop | xargs docker rm
  # add |xargs docker rm to also remove them
  sleep 1

  docker run -d \
    -v $BASE_PATH_/data:/app/data \
    -v $BASE_PATH_/logs:/app/logs \
    -v /home/ankur/nltk_data:/usr/share/nltk_data \
    --env PYTHONUNBUFFERED=1 --env BASE_PATH=/app --env USE_GPU=1 --env PORT=5000 --gpus all --env SERV_ADDR=http://127.0.0.1:5000 \
    --publish 5000:5000 --name=$CONTAINER_NAME \
    -t $IMAGE_NAME:latest
}

run_apache_proxy_server ankur6ue/covid-demo-httpd covid-demo-httpd covid-demo-httpd-dev ~/dev/apps/ML/covid-papers-analysis
run_application_server ankur6ue/covid19-demo covid19-demo ~/dev/apps/ML/covid-papers-analysis

# to see logs: docker ps --filter "name=covid19-demo" -aq | xargs docker logs

