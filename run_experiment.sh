#!/bin/bash

usage(){
    printf "Usage: %s: -a <Algorithm> -x <Experiment Config File Path>\n" $(basename $0) >&2
    exit 2
}

while getopts 'a:x:' OPTION "$@"
do
    case $OPTION in
    a)  Algorithm=$OPTARG
        ;;
    x)  Exp_Config_Path=$OPTARG
        ;;
    ?)  usage
        ;;
    *)  echo "Nothing"
        usage
    esac
done

if [[ -a $Algorithm ]] && [[ -x $Exp_Config_Path ]]
then
    echo "No arguments specified."
    usage
else
    python3 ./experiment/run_experiment.py -a "$Algorithm" -x "$Exp_Config_Path"
    #python3 ./experiment/run_experiment.py -a "MixVPR" -x "/home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment/configs/Config__v2-1_a/v2-1_a_Mahidol_University_1_perspective_images.json"
fi