#!/bin/bash
while true; do
  graphify extract . --directed --wiki --max-concurrency 1
  if [ $? -eq 0 ]; then
    echo "Graphify finished successfully!"
    break
  fi
  echo "Hit rate limit. Waiting 65 seconds for quota to reset..."
  sleep 65
done
