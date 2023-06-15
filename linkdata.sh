#!/bin/bash

IO_root=UNav-IO
original_data_root=~/Data/UNav/Mapping
org_data_dir=$original_data_root/data
org_topomap=$original_data_root/Topomap/Output
org_maps_dir=$org_data_dir/maps
org_floor_plan=$org_data_dir/floor_plan

home=$(pwd)

if [ ! -d "$IO_root" ]; then
    mkdir $IO_root
    echo "$IO_root created."
fi

cd $IO_root
echo "Enter $IO_root directory."

if [ ! -d "data" ]; then
    mkdir data
    echo "$IO_root/data created."
fi

cd data
echo "Enter data directory."

if [ ! -L "destination.json" ]; then
    ln -s "$org_topomap/destination.json" .
fi

for place in $(ls $org_maps_dir)
do
    abs_place=$org_maps_dir/$place
    if [ ! -L "$place" ]; then
        ln -s $abs_place .
        echo "Created a symbolic link to $abs_place."
    else
        echo "Symbolic link to $abs_place exists."
    fi
    
    if [ -d $place ]; then  
        cd $place
        echo "=Current DIR: $(pwd)."

        for building in $(ls $abs_place)
        do
            if [ -d $building ]; then
                abs_building=$abs_place/$building
                cd $building
                echo "==Current DIR: $(pwd)."
                for floor in $(ls $abs_building)
                do
                    if [ -d $floor ]; then
                        abs_floor=$abs_building/$floor
                        cd $floor
                        echo "===Current DIR: $(pwd)."
                        if [ ! -L "boundaries.json" ]; then
                            boundaries="$org_topomap/$place/$building/$floor/boundaries.json"
                            ln -s "$boundaries" .
                            echo "Created a symbolic link to $boundaries."
                        fi 
                        if [ ! -L "topo-map.json" ]; then
                            topomap="$org_topomap/$place/$building/$floor/topo-map.json"
                            ln -s "$topomap" .
                            echo "Created a symbolic link to $topomap."
                        fi 
                        if [ ! -L "floorplan.png" ]; then
                            floorplan="$org_data_dir/floor_plan/$place/$building/$floor.png"
                            ln -s "$floorplan" floorplan.png
                            echo "Created a symbolic link to $floorplan as floorplan.png."
                        fi 
                        cd ..
                    fi
                done
                cd ..
            fi
        done
        cd ..
    fi
done

cd $home
