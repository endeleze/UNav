#!/bin/bash

while getopts 'p:b:f:' OPTION "$@"
do
    case $OPTION in
    p)  Place=$OPTARG
        ;;
    b)  Building=$OPTARG
        ;;
    f)  Floor=$OPTARG
        ;;
    ?)  usage
        ;;
    *)  echo "Nothing"
        usage
    esac
done

source config.sh
source ext-config.sh

if [[ -z $Place ]] && [[ -z $Building ]] && [[ -z $Floor ]]
then
    echo "No arguments specified."
else
    # Must run after finishing the boundary_define.sh
    python  $path_script --topomap_path $topo_out --radius $radius --min_distance $min_distance
fi
