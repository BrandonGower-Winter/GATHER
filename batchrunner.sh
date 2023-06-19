#!/bin/bash

count=0
limit=10
time=60
echo "Batch Info:"
echo "Output Directory: $1"
echo "GATHER Args: $2"
while read l; do
  touch "$1/$l-console.log"
  nohup python3 main.py $2 --seed "$l" >> "$1/$l-console.log" &
  sleep .5
  count=$(ps | grep -c "main.py")
  while [ $count -ge $limit ]; do
      sleep $time
      count=$(ps | grep -c "main.py")  # You may need to change this identifier
  done
done <./resources/seeds_long.list
echo "Done!"