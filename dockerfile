FROM elasticsearch:8.15.3

RUN bin/elasticsearch-plugin install --batch repository-s3