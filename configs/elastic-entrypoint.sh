#!/bin/bash

# Create keystore if it doesn't exist
if [ ! -f config/elasticsearch.keystore ]; then
    echo "Creating elasticsearch.keystore..."
    bin/elasticsearch-keystore create
fi

# Add S3 credentials to keystore
echo "Adding S3 credentials to keystore..."
echo $S3_ACCESS_KEY | bin/elasticsearch-keystore add --stdin s3.client.default.access_key
echo $S3_SECRET_KEY | bin/elasticsearch-keystore add --stdin s3.client.default.secret_key

# Set correct permissions
chown elasticsearch:root config/elasticsearch.keystore

# Execute the original entrypoint with the provided arguments
exec /usr/local/bin/docker-entrypoint.sh "$@"