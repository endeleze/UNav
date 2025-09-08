#!/bin/bash

ssh nattachart.tak@tau.mahidol.ac.th "
        echo Started at $(date)
        source ~/bash_local_unav_server;
        cd DockerImages;
        apptainer exec --nv bank-vis4ion-navigation-demo-v1 bash -c '
        cd /unav;
        source venv/bin/activate;
        cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment;
        #python3 create_groundtruth_v2.py /home/nattachart.tak/Data/experiments/Mapping/data/unav2-data Mahidol_University ICT 1.1_MixVPR;
        #python3 create_groundtruth_v2.py /home/nattachart.tak/Data/experiments/Mapping/data/unav2-data Mahidol_University ICT 1.1_NetVlad;
        #python3 create_groundtruth_v2.py /home/nattachart.tak/Data/experiments/Mapping/data/unav2-data Mahidol_University ICT 1.1_CricaVPR;
        #python3 create_groundtruth_v2.py /home/nattachart.tak/Data/experiments/Mapping/data/unav2-data Mahidol_University ICT 1.1_DinoV2Salad;
        python3 create_groundtruth_v2.py /home/nattachart.tak/Data/experiments/Mapping/data/unav2-data Mahidol_University ICT 1.1_AnyLoc;
        ';
"