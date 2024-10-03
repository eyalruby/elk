FROM elasticsearch:8.12.2

COPY ./elasticsearch.yml /usr/share/elasticsearch/config/elasticsearch.yml

ENV ES_JAVA_OPTS="-Xms512m -Xmx512m"
