#!/bin/bash

# ssh nattachart.tak@tau.mahidol.ac.th "
#         echo "Started at $(date)"
#         screen -dm bash -c \"
#                 cd /home/nattachart.tak/PhD/images;
#                 source ./bash_local_unav_server;
#                 apptainer exec --nv ubuntu2204cudnn1263_mapping_v2 bash -c '
#                 source ./bash_local_unav_server;
#                 cd /home/nattachart.tak/PhD;
#                 source python-env/venv-1/bin/activate;
#                 cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
#                 ./run_mapper_apptainer.sh -a MixVPR -p Mahidol_University -b ICT -f 1.1_MixVPR > run_mapper_apptainer_1.1_MixVPR.log 2>&1;
#                 #./run_mapper_apptainer.sh -a NetVlad -p Mahidol_University -b ICT -f 1.1_NetVlad > run_mapper_apptainer_1.1_NetVlad.log 2>&1;
#                 #./run_mapper_apptainer.sh -a DinoV2Salad -p Mahidol_University -b ICT -f 1.1_DinoV2Salad > run_mapper_apptainer_1.1_DinoV2Salad.log 2>&1;
#                 #./run_mapper_apptainer.sh -a CricaVPR -p Mahidol_University -b ICT -f 1.1_CricaVPR > run_mapper_apptainer_1.1_CricaVPR.log 2>&1;
#                 ';
#         \"
# "

# ssh nattachart.tak@tau.mahidol.ac.th "
#         echo "Started at $(date)"
#         screen -dm bash -c \"
#                 cd /home/nattachart.tak/PhD/images;
#                 source ./bash_local_unav_server;
#                 apptainer exec --nv ubuntu2204cudnn1263_mapping_v2 bash -c '
#                 source ./bash_local_unav_server;
#                 cd /home/nattachart.tak/PhD;
#                 source python-env/venv-1/bin/activate;
#                 cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
#                 #./run_mapper_apptainer.sh -a MixVPR -p Mahidol_University -b ICT -f 1.1_MixVPR > run_mapper_apptainer_1.1_MixVPR.log 2>&1;
#                 ./run_mapper_apptainer.sh -a NetVlad -p Mahidol_University -b ICT -f 1.1_NetVlad > run_mapper_apptainer_1.1_NetVlad.log 2>&1;
#                 #./run_mapper_apptainer.sh -a DinoV2Salad -p Mahidol_University -b ICT -f 1.1_DinoV2Salad > run_mapper_apptainer_1.1_DinoV2Salad.log 2>&1;
#                 #./run_mapper_apptainer.sh -a CricaVPR -p Mahidol_University -b ICT -f 1.1_CricaVPR > run_mapper_apptainer_1.1_CricaVPR.log 2>&1;
#                 ';
#         \"
# "

ssh nattachart.tak@tau.mahidol.ac.th "
        echo "Started at $(date)"
        screen -dm bash -c \"
                cd /home/nattachart.tak/PhD/images;
                source ./bash_local_unav_server;
                apptainer exec --nv ubuntu2204cudnn1263_mapping_v2 bash -c '
                source ./bash_local_unav_server;
                cd /home/nattachart.tak/PhD;
                source python-env/venv-1/bin/activate;
                cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
                #./run_mapper_apptainer.sh -a NetVlad -p Mahidol_University -b ICT -f 1.1_NetVlad > run_mapper_apptainer_1.1_NetVlad.log 2>&1;
                #./run_mapper_apptainer.sh -a DinoV2Salad -p Mahidol_University -b ICT -f 1.1_DinoV2Salad > run_mapper_apptainer_1.1_DinoV2Salad.log 2>&1;
                #./run_mapper_apptainer.sh -a CricaVPR -p Mahidol_University -b ICT -f 1.1_CricaVPR > run_mapper_apptainer_1.1_CricaVPR.log 2>&1;
                ./run_mapper_apptainer.sh -a AnyLoc -p Mahidol_University -b ICT -f 1.1_AnyLoc > run_mapper_apptainer_1.1_AnyLoc.log 2>&1;
                ';
        \"
"

# ssh nattachart.tak@tau.mahidol.ac.th "
#         echo "Started at $(date)"
#         screen -dm bash -c \"
#                 cd /home/nattachart.tak/PhD/images;
#                 source ./bash_local_unav_server;
#                 apptainer exec --nv ubuntu2204cudnn1263_mapping_v2 bash -c '
#                 source ./bash_local_unav_server;
#                 cd /home/nattachart.tak/PhD;
#                 source python-env/venv-1/bin/activate;
#                 cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
#                 #./run_mapper_apptainer.sh -a NetVlad -p Mahidol_University -b ICT -f 1.1_NetVlad > run_mapper_apptainer_1.1_NetVlad.log 2>&1;
#                 #./run_mapper_apptainer.sh -a DinoV2Salad -p Mahidol_University -b ICT -f 1.1_DinoV2Salad > run_mapper_apptainer_1.1_DinoV2Salad.log 2>&1;
#                 ./run_mapper_apptainer.sh -a CricaVPR -p Mahidol_University -b ICT -f 1.1_CricaVPR > run_mapper_apptainer_1.1_CricaVPR.log 2>&1;
#                 ';
#         \"
# "
