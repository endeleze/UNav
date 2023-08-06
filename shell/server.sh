#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script="$CURRENT_DIR/../src/server.py"
host_config="$CURRENT_DIR/../configs/host.yaml"
hloc_config="$CURRENT_DIR/../configs/hloc.yaml"

echo $CURRENT_DIR

usage(){
    printf "Usage: %s: -s <Server YAML File Name>\n" $(basename $0) >&2
        exit 2
}

while getopts 's:' OPTION "$@"
do
    case $OPTION in
    s)  server_yaml=$OPTARG
        ;;
    ?)  usage
        ;;
    *)  echo "Nothing"
        usage
    esac
done

server_yaml=$(echo $server_yaml | sed "s/\.yaml$//g") #Remove .yaml extension if existing

server_config="$CURRENT_DIR/../configs/server/$server_yaml.yaml"

python3 $script -s $server_config -h $host_config -l $hloc_config
