#!/bin/sh
echo "[TARGET] Logstash-Compatible Zabbix Exporter (ubuntu-2204-host Only)"
echo "Target Host: ubuntu-2204-host"
echo "MySQL Host: $MYSQL_HOST"
echo "Logstash URL: $LOGSTASH_URL"

# MySQL connection
export MYSQL_PWD="root"
MYSQL_CMD="mariadb --ssl=0 --host=mysql-server --user=zabbix --database=zabbix"

echo "[CONFIG] Testing connections..."
if $MYSQL_CMD -e "SELECT 1" >/dev/null 2>&1; then
    echo "[OK] MySQL: Connected!"
    
    # Verify target host exists
    HOST_EXISTS=$($MYSQL_CMD -N -e "SELECT COUNT(*) FROM hosts WHERE host = 'ubuntu-2204-host' AND status = 0" 2>/dev/null)
    if [ "$HOST_EXISTS" = "0" ]; then
        echo "[ERROR] Host 'ubuntu-2204-host' not found or disabled in Zabbix"
        exit 1
    else
        echo "[OK] Target host 'ubuntu-2204-host' found and active"
    fi
    
    HISTORY_COUNT=$($MYSQL_CMD -N -e "SELECT COUNT(*) FROM history h JOIN items i ON h.itemid = i.itemid JOIN hosts ON i.hostid = hosts.hostid WHERE h.clock > UNIX_TIMESTAMP(NOW() - INTERVAL 1 HOUR) AND hosts.host = 'ubuntu-2204-host'" 2>/dev/null)
    echo "[DATA] Recent history records for ubuntu-2204-host: ${HISTORY_COUNT:-0}"
else
    echo "[ERROR] MySQL: Failed"
    exit 1
fi

curl -s http://logstash:5044 >/dev/null && echo "[OK] Logstash: Connected!" || echo "[ERROR] Logstash: Failed"

echo "[START] Starting filtered data export (ubuntu-2204-host only)..."
sleep 10

cycle=1
while true; do
    echo "=========================================="
    echo "[CYCLE] Export Cycle #$cycle - $(date)"
    echo "[TARGET] Extracting data ONLY from ubuntu-2204-host"
    
    # Export from history table (float values) - FIXED
    $MYSQL_CMD -N -e "
    SELECT 
        h.itemid,
        h.clock,
        h.value,
        h.ns,
        REPLACE(REPLACE(REPLACE(REPLACE(i.key_, '\"', ''), CHAR(10), ' '), CHAR(13), ' '), CHAR(9), ' ') as clean_key,
        REPLACE(REPLACE(REPLACE(REPLACE(i.name, '\"', ''), CHAR(10), ' '), CHAR(13), ' '), CHAR(9), ' ') as clean_name,
        COALESCE(NULLIF(REPLACE(REPLACE(i.units, '\"', ''), CHAR(10), ' '), ''), 'none') as clean_units,
        REPLACE(REPLACE(hosts.host, '\"', ''), ' ', '_') as clean_host,
        'history' as table_type
    FROM history h
    JOIN items i ON h.itemid = i.itemid
    JOIN hosts ON i.hostid = hosts.hostid
    WHERE h.clock > UNIX_TIMESTAMP(NOW() - INTERVAL 5 MINUTE)
      AND i.status = 0
      AND hosts.status = 0
      AND hosts.host = 'ubuntu-2204-host'
    ORDER BY h.clock DESC
    LIMIT 5
    " 2>/dev/null | while IFS=$'\t' read itemid clock value ns item_key item_name units hostname table_type; do
        
        if [ -n "$itemid" ] && [ -n "$value" ]; then
            # Clean fields
            clean_key=$(echo "$item_key" | tr -d '"' | tr '\n\r\t' ' ' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
            clean_name=$(echo "$item_name" | tr -d '"' | tr '\n\r\t' ' ' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
            clean_units=$(echo "$units" | tr -d '"' | tr '\n\r\t' ' ' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
            clean_host=$(echo "$hostname" | tr -d '"' | tr ' ' '_')
            
            # Create JSON payload with host validation
            json_payload="{\"type\":\"zabbix-metrics\",\"itemid\":$itemid,\"clock\":$clock,\"value\":$value,\"ns\":$ns,\"item_key\":\"$clean_key\",\"item_name\":\"$clean_name\",\"units\":\"$clean_units\",\"hostname\":\"$clean_host\",\"table_type\":\"$table_type\",\"export_timestamp\":$(date +%s),\"target_host\":\"ubuntu-2204-host\"}"
            
            # Send to Logstash
            response=$(curl -s -w "%{http_code}" -X POST "$LOGSTASH_URL" \
                       -H "Content-Type: application/json" \
                       -d "$json_payload" -o /dev/null)
            
            if [ "$response" = "200" ]; then
                echo "[OK] [ubuntu-2204-host] $clean_key = $value $clean_units"
            else
                echo "[FAIL] Failed: $clean_key (HTTP $response)"
            fi
        fi
    done
    
    # Export from history_uint table - FIXED
    $MYSQL_CMD -N -e "
    SELECT 
        h.itemid,
        h.clock,
        h.value,
        h.ns,
        REPLACE(REPLACE(REPLACE(REPLACE(i.key_, '\"', ''), CHAR(10), ' '), CHAR(13), ' '), CHAR(9), ' ') as clean_key,
        REPLACE(REPLACE(REPLACE(REPLACE(i.name, '\"', ''), CHAR(10), ' '), CHAR(13), ' '), CHAR(9), ' ') as clean_name,
        COALESCE(NULLIF(REPLACE(REPLACE(i.units, '\"', ''), CHAR(10), ' '), ''), 'none') as clean_units,
        REPLACE(REPLACE(hosts.host, '\"', ''), ' ', '_') as clean_host,
        'history_uint' as table_type
    FROM history_uint h
    JOIN items i ON h.itemid = i.itemid
    JOIN hosts ON i.hostid = hosts.hostid
    WHERE h.clock > UNIX_TIMESTAMP(NOW() - INTERVAL 5 MINUTE)
      AND i.status = 0
      AND hosts.status = 0
      AND hosts.host = 'ubuntu-2204-host'
    ORDER BY h.clock DESC
    LIMIT 3
    " 2>/dev/null | while IFS=$'\t' read itemid clock value ns item_key item_name units hostname table_type; do
        
        if [ -n "$itemid" ] && [ -n "$value" ]; then
            clean_key=$(echo "$item_key" | tr -d '"' | tr '\n\r\t' ' ' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
            clean_name=$(echo "$item_name" | tr -d '"' | tr '\n\r\t' ' ' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
            clean_units=$(echo "$units" | tr -d '"' | tr '\n\r\t' ' ' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
            clean_host=$(echo "$hostname" | tr -d '"' | tr ' ' '_')
            
            json_payload="{\"type\":\"zabbix-metrics\",\"itemid\":$itemid,\"clock\":$clock,\"value\":$value,\"ns\":$ns,\"item_key\":\"$clean_key\",\"item_name\":\"$clean_name\",\"units\":\"$clean_units\",\"hostname\":\"$clean_host\",\"table_type\":\"$table_type\",\"export_timestamp\":$(date +%s),\"target_host\":\"ubuntu-2204-host\"}"
            
            response=$(curl -s -w "%{http_code}" -X POST "$LOGSTASH_URL" \
                       -H "Content-Type: application/json" \
                       -d "$json_payload" -o /dev/null)
            
            if [ "$response" = "200" ]; then
                echo "[OK] [ubuntu-2204-host] $clean_key = $value $clean_units"
            else
                echo "[FAIL] Failed: $clean_key (HTTP $response)"
            fi
        fi
    done
    
    cycle=$((cycle + 1))
    echo "[DONE] Cycle complete - ubuntu-2204-host only. Waiting $EXPORT_INTERVAL seconds..."
    sleep "$EXPORT_INTERVAL"
done
