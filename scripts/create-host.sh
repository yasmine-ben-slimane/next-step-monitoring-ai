#!/bin/sh

echo "Installing dependencies..."
apk add --no-cache curl jq

echo "Waiting for services..."
sleep 90

echo "Testing API connection..."
if ! curl -s http://zabbix-web:8080/api_jsonrpc.php > /dev/null; then
    echo "API not ready, waiting more..."
    sleep 60
fi

echo "Authenticating..."
AUTH_TOKEN=$(curl -s -X POST http://zabbix-web:8080/api_jsonrpc.php \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "user.login", "params": {"user": "Admin", "password": "zabbix"}, "id": 1}' | jq -r ".result")

if [ "$AUTH_TOKEN" = "null" ]; then
    echo "Authentication failed"
    exit 1
fi

echo "Checking existing host..."
HOST_EXISTS=$(curl -s -X POST http://zabbix-web:8080/api_jsonrpc.php \
  -H "Content-Type: application/json" \
  -d "{\"jsonrpc\": \"2.0\", \"method\": \"host.get\", \"params\": {\"filter\": {\"host\": [\"ubuntu-2204-host\"]}}, \"auth\": \"$AUTH_TOKEN\", \"id\": 2}" | jq -r ".result | length")

if [ "$HOST_EXISTS" != "0" ]; then
    echo "Host already exists"
    exit 0
fi

echo "Getting group..."
GROUP_ID=$(curl -s -X POST http://zabbix-web:8080/api_jsonrpc.php \
  -H "Content-Type: application/json" \
  -d "{\"jsonrpc\": \"2.0\", \"method\": \"hostgroup.get\", \"params\": {\"filter\": {\"name\": [\"Linux servers\"]}}, \"auth\": \"$AUTH_TOKEN\", \"id\": 3}" | jq -r ".result[0].groupid")

if [ "$GROUP_ID" = "null" ]; then
    echo "Creating group..."
    GROUP_ID=$(curl -s -X POST http://zabbix-web:8080/api_jsonrpc.php \
      -H "Content-Type: application/json" \
      -d "{\"jsonrpc\": \"2.0\", \"method\": \"hostgroup.create\", \"params\": {\"name\": \"Linux servers\"}, \"auth\": \"$AUTH_TOKEN\", \"id\": 4}" | jq -r ".result.groupids[0]")
fi

echo "Getting template..."
TEMPLATE_ID=$(curl -s -X POST http://zabbix-web:8080/api_jsonrpc.php \
  -H "Content-Type: application/json" \
  -d "{\"jsonrpc\": \"2.0\", \"method\": \"template.get\", \"params\": {\"filter\": {\"host\": [\"Linux by Zabbix agent\"]}}, \"auth\": \"$AUTH_TOKEN\", \"id\": 5}" | jq -r ".result[0].templateid")

if [ "$TEMPLATE_ID" = "null" ]; then
    echo "Template not found"
    exit 1
fi

echo "Creating host..."
CREATE_RESULT=$(curl -s -X POST http://zabbix-web:8080/api_jsonrpc.php \
  -H "Content-Type: application/json" \
  -d "{\"jsonrpc\": \"2.0\", \"method\": \"host.create\", \"params\": {\"host\": \"ubuntu-2204-host\", \"name\": \"Ubuntu VM\", \"interfaces\": [{\"type\": 1, \"main\": 1, \"useip\": 0, \"ip\": \"\", \"dns\": \"zabbix-agent\", \"port\": \"10050\"}], \"groups\": [{\"groupid\": \"$GROUP_ID\"}], \"templates\": [{\"templateid\": \"$TEMPLATE_ID\"}]}, \"auth\": \"$AUTH_TOKEN\", \"id\": 6}")

HOST_ID=$(echo "$CREATE_RESULT" | jq -r ".result.hostids[0]")

if [ "$HOST_ID" != "null" ]; then
    echo "SUCCESS: Host created with ID: $HOST_ID"
else
    echo "FAILED: Could not create host"
    echo "$CREATE_RESULT"
fi

echo "Done!"
