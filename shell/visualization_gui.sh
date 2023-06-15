#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script="$CURRENT_DIR/../src/visualization_gui.py"
hloc_config="$CURRENT_DIR/../configs/hloc.yaml"

usage(){
    printf "Usage: %s: -v <Visualization YAML File Name>\n" $(basename $0) >&2 
        exit 2
}

while getopts 'v:' OPTION "$@"
do
    case $OPTION in
    s)  visual_yaml=$OPTARG
        ;;  
    ?)  usage
        ;;  
    *)  echo "Nothing"
        usage
    esac
done

visual_yaml=$(echo $visual_yaml | sed "s/\.yaml$//g") #Remove .yaml extension if existing

visual_config="$CURRENT_DIR/../configs/visualization/$visual_yaml.yaml"

python $script -v $visual_config -l $hloc_config
