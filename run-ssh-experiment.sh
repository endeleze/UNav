#!/bin/bash

ssh nattachart.tak@tau.mahidol.ac.th "
        echo Started at $(date)
        screen -dm bash -c \"
                source ~/bash_local_unav_server;
                cd DockerImages;
                apptainer exec --nv bank-vis4ion-navigation-demo-v1 bash -c '
                cd /unav;
                source venv/bin/activate;
                cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
                ./run_experiment.sh -a \"MixVPR\" -x \"/home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment/configs/Config__v2-1_c/config.json\" > run_experiment_Config__v2-1_c_MixVPR.log 2>&1;
                ';
        \"
"

# ssh nattachart.tak@tau.mahidol.ac.th "
#         echo Started at $(date)
#         screen -dm bash -c \"
#                 source ~/bash_local_unav_server;
#                 cd DockerImages;
#                 apptainer exec --nv bank-vis4ion-navigation-demo-v1 bash -c '
#                 cd /unav;
#                 source venv/bin/activate;
#                 cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
#                 ./run_experiment.sh -a \"DinoV2Salad\" -x \"/home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment/configs/Config__v2-1_c/config.json\" > run_experiment_Config__v2-1_c_DinoV2Salad.log 2>&1;
#                 ';
#         \"
# "

# ssh nattachart.tak@tau.mahidol.ac.th "
#         echo Started at $(date)
#         screen -dm bash -c \"
#                 source ~/bash_local_unav_server;
#                 cd DockerImages;
#                 apptainer exec --nv bank-vis4ion-navigation-demo-v1 bash -c '
#                 cd /unav;
#                 source venv/bin/activate;
#                 cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
#                 ./run_experiment.sh -a \"NetVlad\" -x \"/home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment/configs/Config__v2-1_c/config.json\" > run_experiment_Config__v2-1_c_NetVlad.log 2>&1;
#                 ';
#         \"
# "

# ssh nattachart.tak@tau.mahidol.ac.th "
#         echo Started at $(date)
#         screen -dm bash -c \"
#                 source ~/bash_local_unav_server;
#                 cd DockerImages;
#                 apptainer exec --nv bank-vis4ion-navigation-demo-v1 bash -c '
#                 cd /unav;
#                 source venv/bin/activate;
#                 cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
#                 ./run_experiment.sh -a \"CricaVPR\" -x \"/home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment/configs/Config__v2-1_c/config.json\" > run_experiment_Config__v2-1_c_CricaVPR.log 2>&1;
#                 ';
#         \"
# "

# ssh nattachart.tak@tau.mahidol.ac.th "
#         echo Started at $(date)
#         screen -dm bash -c \"
#                 source ~/bash_local_unav_server;
#                 cd DockerImages;
#                 apptainer exec --nv bank-vis4ion-navigation-demo-v1 bash -c '
#                 cd /unav;
#                 source venv/bin/activate;
#                 cd /home/nattachart.tak/PhD/Trial_New_UNav/UNav;
#                 ./run_experiment.sh -a \"AnyLoc\" -x \"/home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment/configs/Config__v2-1_c/config.json\" > run_experiment_Config__v2-1_c_AnyLoc.log 2>&1;
#                 ';
#         \"
# "