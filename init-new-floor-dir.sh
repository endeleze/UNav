#!/bin/bash

DATA_FINAL_BUILDING=/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data/Mahidol_University/ICT

REF_FLOOR=1.1_Initial_Files

#NEW_FLOOR=1.1_MixVPR
#NEW_FLOOR=1.1_NetVlad
#NEW_FLOOR=1.1_CricaVPR
#NEW_FLOOR=1.1_DinoV2Salad
NEW_FLOOR=1.1_AnyLoc

REF_FLOOR_PATH=$DATA_FINAL_BUILDING/$REF_FLOOR
NEW_FLOOR_PATH=$DATA_FINAL_BUILDING/$NEW_FLOOR

if [ ! -d "$NEW_FLOOR_PATH" ]; then
    mkdir "$NEW_FLOOR_PATH"
    if [ $? -eq 0 ]; then
        echo "===== Created '$NEW_FLOOR_PATH'."
    fi
else
    echo "!==== '$NEW_FLOOR_PATH' already exists."
fi

if [ ! -f "$NEW_FLOOR_PATH/$NEW_FLOOR.mp4" ]; then
    ln -s "$REF_FLOOR_PATH/$REF_FLOOR.mp4" "$NEW_FLOOR_PATH/$NEW_FLOOR.mp4"
    if [ $? -eq 0 ]; then
        echo "===== Created '$NEW_FLOOR_PATH/$NEW_FLOOR.mp4' linked to '$REF_FLOOR_PATH/$REF_FLOOR.mp4'."
    fi
else
    echo "!==== '$NEW_FLOOR_PATH/$NEW_FLOOR.mp4' already exists."
fi

if [ ! -f "$NEW_FLOOR_PATH/floorplan.png" ]; then
    cp "$REF_FLOOR_PATH/floorplan.png" "$NEW_FLOOR_PATH/floorplan.png"
    if [ $? -eq 0 ]; then
        echo "===== Created '$NEW_FLOOR_PATH/floorplan.png'."
    fi
else
    echo "!==== '$NEW_FLOOR_PATH/floorplan.png' already exists."
fi

if [ ! -d "$NEW_FLOOR_PATH/stella_vslam_dense" ]; then
    ln -s "$REF_FLOOR_PATH/stella_vslam_dense" "$NEW_FLOOR_PATH/stella_vslam_dense"
    if [ $? -eq 0 ]; then
        echo "===== Created '$NEW_FLOOR_PATH/stella_vslam_dense' linked to '$REF_FLOOR_PATH/stella_vslam_dense'."
    fi
else
    echo "!==== '$NEW_FLOOR_PATH/stella_vslam_dense' already exists."
fi

if [ ! -d "$NEW_FLOOR_PATH/perspectives" ]; then
    ln -s "$REF_FLOOR_PATH/perspectives" "$NEW_FLOOR_PATH/perspectives"
    if [ $? -eq 0 ]; then
        echo "===== Created '$NEW_FLOOR_PATH/perspectives' linked to '$REF_FLOOR_PATH/perspectives'."
    fi
else
    echo "!==== '$NEW_FLOOR_PATH/perspectives' already exists."
fi

if [ ! -d "$NEW_FLOOR_PATH/colmap_sfm" ]; then
    cp -r "$REF_FLOOR_PATH/colmap_sfm" "$NEW_FLOOR_PATH/colmap_sfm"
    if [ $? -eq 0 ]; then
        echo "===== Created '$NEW_FLOOR_PATH/colmap_sfm'."
    fi
else
    echo "!==== '$NEW_FLOOR_PATH/colmap_sfm' already exists."
fi

if [ ! -d "$NEW_FLOOR_PATH/aligner" ]; then
    cp -r "$REF_FLOOR_PATH/aligner" "$NEW_FLOOR_PATH/aligner"
    if [ $? -eq 0 ]; then
        echo "===== Created '$NEW_FLOOR_PATH/aligner'."
    fi
else
    echo "!==== '$NEW_FLOOR_PATH/aligner' already exists."
fi