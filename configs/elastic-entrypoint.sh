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

# Import SSL certificate for S3 endpoint
if [ ! -f /usr/share/elasticsearch/config/certs/public.crt ]; then
    echo "Fetching SSL certificate for S3 endpoint..."
    openssl s_client -connect $S3_ENDPOINT:443 < /dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > /usr/share/elasticsearch/config/certs/public.crt

    echo "Importing SSL certificate into Elasticsearch keystore..."
    /usr/share/elasticsearch/jdk/bin/keytool -import -trustcacerts \
        -alias s3-cert \
        -keystore /usr/share/elasticsearch/config/certs/s3-truststore.jks \
        -file /usr/share/elasticsearch/config/certs/public.crt -storepass changeit -noprompt
fi

# Set correct permissions
chown elasticsearch:root config/elasticsearch.keystore

# Execute the original entrypoint with the provided arguments
exec /usr/local/bin/docker-entrypoint.sh "$@"