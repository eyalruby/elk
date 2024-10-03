
# Wait for Elasticsearch to be ready
until curl -k -u elastic:$ELASTIC_PASSWORD https://elasticsearch-0:9200 -s; do
  echo "Waiting for Elasticsearch to start..."
  sleep 5
done

# Set the password for the kibana_system user
curl -k -u elastic:$ELASTIC_PASSWORD -X POST "https://elasticsearch-0:9200/_security/user/kibana_system/_password" \
  -H "Content-Type: application/json" -d'
{
  "password" : "'"$KIBANA_PASSWORD"'"
}'

# Create a Kibana user with UI access
curl -k -u elastic:$ELASTIC_PASSWORD -X POST "https://elasticsearch-0:9200/_security/user/kibana_user" \
  -H "Content-Type: application/json" -d'
{
  "password" : "'"$KIBANA_PASSWORD"'",
  "roles" : [ "kibana_admin" ],
  "full_name" : "Kibana User",
  "email" : "user@example.com"
}'

echo "Kibana user created successfully!"
