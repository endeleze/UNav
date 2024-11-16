import faulthandler
import numpy as np
import logging
from feature.global_extractor import Global_Extractors
from feature.local_extractor import Local_extractor
from feature.local_matcher import Local_matcher
from third_party.torchSIFT.src.torchsift.ransac.ransac import ransac
from third_party.torchSIFT.src.torchsift.ransac.matcher import match
from .implicit_distortion_model import coarse_pose,pose_multi_refine
import torch
import cv2
from time import time

class Hloc():
    device='cuda' if torch.cuda.is_available() else "cpu"
    def __init__(self, root, map_data, config):
        self.config=config['hloc']
        self.batch_mode=self.config['batch_mode']
        self.thre=self.config['ransac_thre']
        self.match_type=self.config['match_type']
        self.rot_base=map_data['rot_base']
        self.T=map_data['T']
        self.db_desc=map_data['global_descriptor']
        self.keyframe_name=map_data['keyframe_name']
        feature_configs=config['feature']
        global_extractor = Global_Extractors(root,feature_configs['global'])
        self.global_extractor=global_extractor.get()
        local_feature = Local_extractor(feature_configs['local'])
        self.local_feature_extractor = local_feature.extractor()
        self.local_feature_matcher=Local_matcher(map_data['database_name'],map_data['local_descriptor'],map_data['registered_feature'],**feature_configs)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        self.list_2d, self.list_3d, self.initial_poses, self.pps = [], [], [], []
        self.last_time=time()

    def global_retrieval(self,image):
        """
        Global Images Retrieval:
            Find the topk closest database images of input query image
        """
        self.query_desc = self.global_extractor(image)
        # print(self.query_desc)
        # exit()
        sim = torch.einsum('id,jd->ij', self.query_desc, self.db_desc)
        topk = torch.topk(sim, self.config['retrieval_num'], dim=1).indices.cpu().numpy()
        return topk

    def feature_matching_lightglue_batch(self,image,topk):
        """
        Local Feature Matching:
            Match the local features between query image and retrieved database images
        """
        with torch.inference_mode():  # Use torch.no_grad during inference
            feats0 = self.local_feature_extractor(image)
        self.retrived_image_index=[]
        pts0_list,pts1_list,lms_list,max_len=self.local_feature_matcher.lightglue_batch(self,topk[0], feats0)
        self.retrived_image_index=torch.tensor(self.retrived_image_index).to(self.device)
        return pts0_list,pts1_list,lms_list,max_len

    def feature_matching_lightglue(self,image,topk):
        """
        Local Feature Matching:
            Match the local features between query image and retrieved database images
        """
        with torch.inference_mode():  # Use torch.no_grad during inference
            feats0 = self.local_feature_extractor(image)
        pts0_list,pts1_list,lms_list=[],[],[]
        max_len=0
        self.retrived_image_index=[]
        for i in topk[0]:
            pts0,pts1,lms=self.local_feature_matcher.lightglue(i, feats0)
            feat_inliner_size=pts0.shape[0]
            if feat_inliner_size>self.thre:
                pts0_list.append(pts0)
                pts1_list.append(pts1)
                lms_list.append(lms)
                self.retrived_image_index.append(i)
                if feat_inliner_size>max_len:
                    max_len=feat_inliner_size
            del pts0,pts1,lms
        self.retrived_image_index=torch.tensor(self.retrived_image_index).to(self.device)
        del self.query_desc, feats0
        torch.cuda.empty_cache()
        return pts0_list,pts1_list,lms_list,max_len
    
    def feature_matching_superglue(self,image,topk):
        """
        Local Feature Matching:
            Match the local features between query image and retrieved database images
        """
        with torch.inference_mode():  # Use torch.no_grad during inference
            feats0 = self.local_feature_extractor(image)
        pts0_list,pts1_list,lms_list=[],[],[]
        max_len=0
        self.retrived_image_index=[]
        for i in topk[0]:
            pts0,pts1,lms=self.local_feature_matcher.superglue(i, feats0)
            feat_inliner_size=pts0.shape[0]
            if feat_inliner_size>self.thre:
                pts0_list.append(pts0)
                pts1_list.append(pts1)
                lms_list.append(lms)
                self.retrived_image_index.append(i)
                if feat_inliner_size>max_len:
                    max_len=feat_inliner_size
        self.retrived_image_index=torch.tensor(self.retrived_image_index).to(self.device)
        del self.query_desc, feats0
        torch.cuda.empty_cache()
        return pts0_list,pts1_list,lms_list,max_len

    def geometric_verification(self,pts0_list,pts1_list,lms_list,max_len):
        """
        Geometric verification:
            Apply geometric verification between query and database images
        """
        batch_size=len(pts0_list)
        pts0=torch.empty((batch_size,max_len,2),dtype=float)
        pts1=torch.empty((batch_size,max_len,2),dtype=float)
        lms=torch.empty((batch_size,max_len,3),dtype=float)
        mask=torch.zeros((batch_size,max_len,max_len))
        for i in range(batch_size):
            inliner_size=len(pts0_list[i])
            pts0[i,:inliner_size,:]=torch.from_numpy(pts0_list[i])
            pts1[i,:inliner_size,:]=torch.from_numpy(pts1_list[i])
            lms[i,:inliner_size,:]=torch.from_numpy(lms_list[i])
            mask[i,:inliner_size,:inliner_size]=torch.ones((inliner_size,inliner_size))
        pts0,pts1,lms,mask=pts0.to(self.device),pts1.to(self.device),lms.to(self.device),mask.to(self.device)
        try:
            _,inliners,_=ransac(pts0,pts1,mask)
        except:
            del pts0,pts1,lms,mask
            torch.cuda.empty_cache()
            return torch.tensor([]),None

        diag_masks = torch.diagonal(inliners, dim1=-2, dim2=-1)

        # Calculate sizes for each item in the batch
        sizes = diag_masks.sum(-1)

        # Filtering based on the threshold
        valid_indices = sizes > self.thre

        # Apply thresholding
        diag_masks = diag_masks[valid_indices]
        pts0 = pts0[valid_indices]
        lms = lms[valid_indices]

        self.retrived_image_index=self.retrived_image_index[valid_indices]
        # Masking pts0, pts1, and lms
        masked_pts0 = [pts0[i][diag_masks[i]] for i in range(pts0.size(0))]
        masked_lms = [lms[i][diag_masks[i]] for i in range(lms.size(0))]
        
        del pts0,pts1,lms,mask
        torch.cuda.empty_cache()
        if len(masked_pts0)>0:
            return torch.cat(masked_pts0),torch.cat(masked_lms)
        else:
            return torch.tensor([]),torch.tensor([])

    def nvs_ot(self,image,topk):
        feats0 = self.local_feature_extractor(image)
        fts1_list,desc1_list,lms=[],[],[]
        max_len=feats0['keypoints'].shape[0]
        for i in topk[0]:
            feat1=self.local_feature_matcher.hfile_local[self.local_feature_matcher.db_name[i]]
            registered_feature = self.local_feature_matcher.registered_feature[i]
            lms.append({key: value for key, value in zip(registered_feature[0], registered_feature[1])})
            fts1_list.append(torch.tensor(np.array(feat1['keypoints'])))
            desc1_list.append(torch.transpose(torch.tensor(np.array(feat1['descriptors'])),0,1))
            size=feat1['keypoints'].shape[0]
            if max_len<size:
                max_len=size

        k=len(topk[0])
        fts0_list,desc0_list=[],[]
        ft0=torch.tensor(feats0['keypoints'])
        des0=torch.transpose(torch.tensor(feats0['descriptors']),0,1).unsqueeze(0)
        m=des0.size(2)
        fts0_ = ft0.expand(k, -1, -1)
        desc0_=des0.expand(k, -1, -1)
        fts0 = torch.zeros(k, max_len, 2)
        desc0=torch.zeros(k, max_len, m)
        fts0_mask=torch.zeros(k, max_len, 2)
        desc0_mask=torch.zeros(k, max_len, m)

        fts0[:,:feats0['keypoints'].shape[0]]=fts0_
        desc0[:,:feats0['keypoints'].shape[0]]=desc0_
        fts0_mask[:,:feats0['keypoints'].shape[0]]=1
        desc0_mask[:,:feats0['keypoints'].shape[0]]=1

        fts1=torch.zeros((fts0.size(0),max_len,2))
        desc1=torch.zeros((desc0.size(0),max_len,m))
        fts1_mask=torch.zeros(k, max_len, 2)
        desc1_mask=torch.zeros(k, max_len, m)
        print(max_len)
        for i in range(fts1.size(0)):
            fts=fts1_list[i]
            desc=desc1_list[i]
            fts1[i,:fts.size(0),:]=fts
            fts1_mask[i,:fts.size(0),:]=1
            desc1[i,:desc.size(0),:]=desc
            desc1_mask[i,:desc.size(0),:]=1

        best_model, inliers, best_errors=match(fts0.double(),desc0,fts1.double(),desc1)

        self.retrived_image_index=[]

        for k,best_error in enumerate(best_errors):
            mask = torch.isnan(best_error) == False
            valid_values = best_error[mask]
            error=sum(valid_values)/sum(fts1_mask[k])[0]
            if error<3000:
                self.retrived_image_index.append(k)
                # inlier=inliers[k]
                print(len(valid_values))
            # print(sum(valid_values))
            # sum_num=0
            # for i in k:
            #     mask = torch.isnan(i) == False
            #     valid_values = i[mask]
            #     sum_num+=sum(valid_values<1.6)
            # print('matched:%d'%sum_num)
        # Move tensors to CPU after use and delete their references
        del fts0, desc0, fts1, desc1  
        torch.cuda.empty_cache()  # Empty the GPU cache
        exit()
        # Calculate sizes for each item in the batch
        sizes = diag_masks.sum(-1)

        # Filtering based on the threshold
        valid_indices = sizes > self.thre

        # Apply thresholding
        diag_masks = diag_masks[valid_indices]
        pts0 = pts0[valid_indices]
        print(pts0)
        # for k in range(inliner.size(0)):
        #     mask=inliner[k]
        #     ft0=fts0[k]
        #     ft1=fts1[k]

    def pnp(self,image,feature2D,landmark3D):
        """
        Start Perspective-n-points:
            Estimate the current location using implicit distortion model
        """
        faulthandler.enable()
        if feature2D.size()[0]>0:
            height, width, _ = image.shape
            print("==================== feature2D - {0}:{2}, landmark3D - {1}:{3}  =================".format(feature2D.size(), landmark3D.size(), type(feature2D), type(landmark3D)))
            feature2D, landmark3D=feature2D.cpu().numpy(),landmark3D.cpu().numpy()
            #feature2D, landmark3D=np.ascontiguousarray(np.copy(feature2D.cpu().numpy().astype(np.float32))),np.ascontiguousarray(np.copy(landmark3D.cpu().numpy().astype(np.float32)))
            out, p2d_inlier, p3d_inlier = coarse_pose(feature2D, landmark3D, np.array([width / 2, height / 2]))
            self.list_2d.append(p2d_inlier)
            self.list_3d.append(p3d_inlier)
            self.matched_2D,self.matched_3D=p2d_inlier,p3d_inlier
            self.initial_poses.append(out['pose'])
            self.pps.append(out['pp'])
            if len(self.list_2d) > self.config['implicit_num']:
                self.list_2d.pop(0)
                self.list_3d.pop(0)
                self.initial_poses.pop(0)
                self.pps.pop(0)
            pose = pose_multi_refine(self.list_2d, self.list_3d, self.initial_poses, self.pps,self.rot_base,self.T)
        else:
            pose =None
            self.logger.warning("!!!Cannot localize at this point, please take some steps or turn around!!!")
        return pose

    def get_location(self, image):
        self.logger.info("Start image retrieval")
        topk=self.global_retrieval(image)

        if self.match_type=='superglue':
            self.logger.info("Matching local feature")
            pts0_list,pts1_list,lms_list,max_len=self.feature_matching_superglue(image,topk)
            self.logger.info("Start geometric verification")
            feature2D,landmark3D=self.geometric_verification(pts0_list, pts1_list, lms_list,max_len)

        elif self.match_type=='lightglue':
            self.logger.info("Matching local feature")
            if self.batch_mode:
                pts0_list,pts1_list,lms_list,max_len=self.feature_matching_lightglue_batch(image,topk)
            else:
                pts0_list,pts1_list,lms_list,max_len=self.feature_matching_lightglue(image,topk)

            self.logger.info("Start geometric verification")
            feature2D,landmark3D=self.geometric_verification(pts0_list, pts1_list, lms_list,max_len)

        elif self.match_type=='nvs':
            self.logger.info("Doing NVS+OT")
            feature2D,landmark3D=self.nvs_ot(image,topk)

        self.logger.info("Estimate the camera pose using PnP algorithm")
        pose=self.pnp(image,feature2D,landmark3D)

        return pose
