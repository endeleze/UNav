usage(){
    printf "Usage: %s: -p <Images DIR Path>\n" $(basename $0) >&2
    exit 2
}

# DATA_TEMP_ROOT="/home/nattachart.tak/Data/experiments/Magnetometer/data"

while getopts 'p:' OPTION "$@"
do
    case $OPTION in
    p)  IMG_DIR=$OPTARG
        ;;
    ?)  usage
        ;;
    *)  echo "Nothing"
        usage
    esac
done

echo $IMG_DIR

python3 -m unav.perspective_img_slicer $IMG_DIR