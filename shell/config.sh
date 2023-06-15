#!/bin/bash

username=nattachart.tak
base_dir=/home/$username

data_root=$base_dir/updates-unav/mahidol-branch/UNav/UNav-IO/data
work_path=$base_dir/updates-unav/mahidol-branch/UNav

path_script=$work_path/Path_finder_waypoints.py

rate=30
#rate=24

# $Place, $Building, $Floor are expected to be specified before this script is sourced.

usage(){
    printf "Usage: %s: -p <Place> -b <Building> -f <Floor>\n" $(basename $0) >&2
    exit 2
}

usage_gpu(){
    printf "Usage: %s: -p <Place> -b <Building> -f <Floor> -g <GPU Index>\n" $(basename $0) >&2
    exit 2
}

#echo "$Place, $Building, $Floor"
