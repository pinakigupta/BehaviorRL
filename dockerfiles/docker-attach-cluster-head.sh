#!/bin/bash
# If Ray cluster is launched this will log on to the docker node on cluster head

ray attach --cluster-name BehaviorRL ../Ray-Cluster.yaml << EOF
docker attach $(docker ps -aqf "name=ray_docker")
#docker exec -it $(docker ps -aqf "name=ray_docker") /bin/bash
EOF




