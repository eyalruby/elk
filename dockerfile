FROM elasticsearch:8.15.3

# Copy the plugin to the container
COPY plugins/analysis-hebrew-8.15.3-commercial-e6bd35b79.zip /tmp/

# Install the plugin
RUN bin/elasticsearch-plugin install file:///tmp/analysis-hebrew-8.15.3-commercial-e6bd35b79.zip --batch 