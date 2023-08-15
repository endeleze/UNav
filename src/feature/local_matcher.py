from feature.local_extractor import Local_extractor
from third_party.local_feature.LightGlue.lightglue.utils import load_image, match_pair
import torch
import numpy as np

class Local_matcher():
    device='cuda' if torch.cuda.is_available() else "cpu"
    def __init__(self,db_name, hfile_local, registered_feature,**feature_configs):
        local_feature = Local_extractor(feature_configs['local'])
        self.local_feature_matcher = local_feature.matcher().to(self.device)
        self.db_name=db_name
        self.hfile_local=hfile_local
        self.registered_feature=registered_feature

    def match_filter(self, matches, feats0, feats1, feature_dict, index_list):
        pts0 = np.empty((len(matches), 2))
        pts1 = np.empty((len(matches), 2))
        lms = np.empty((len(matches),3))
        counter = 0
        for n, m in enumerate(matches):
            if m != -1 and m in index_list:
                pts0[counter] = feats0['keypoints'][n]
                pts1[counter] = feats1['keypoints'][m]
                lms[counter] = feature_dict[m]
                counter += 1
        pts0 = pts0[:counter]
        pts1 = pts1[:counter]
        lms = lms[:counter]
        return pts0, pts1, lms

    def lightglue_batch(self,parent, topk, feats0):
        batch_size=len(topk)
        mini_batch_size=15
        batch_list=[]
        b=batch_size
        while b>0:
            if b<mini_batch_size:
                batch_list.append(b)
                b=0
            else:
                batch_list.append(mini_batch_size)
                b-=mini_batch_size

        pts0_list,pts1_list,lms_list=[],[],[]

        max_dim=0
        index=0
        for batch in batch_list:
            pred={}
            feats1=[]
            pred['descriptors0']=torch.tensor(np.repeat(feats0['descriptors'].__array__()[np.newaxis, :,:], batch, axis=0)).to(self.device)
            pred['image_size0']=torch.tensor(np.repeat(feats0['image_size'].__array__()[np.newaxis, :], batch, axis=0)).to(self.device)
            pred['keypoints0']=torch.tensor(np.repeat(feats0['keypoints'].__array__()[np.newaxis, :,:], batch, axis=0)).to(self.device)
            pred['keypoint_scores0']=torch.tensor(np.repeat(feats0['scores'].__array__()[np.newaxis, :], batch, axis=0)).to(self.device)
            max_len=0
            for ind,i in enumerate(topk[index:index+batch]):
                feats1.append(self.hfile_local[self.db_name[i]])
                desc_dim,current_len=feats1[-1]['descriptors'].__array__().shape
                if ind==0:
                    image_size=feats1[-1]['image_size'].__array__()
                if max_len<current_len:
                    max_len=current_len
            index+=batch
            if max_dim<max_len:
                max_dim=max_len
            descriptors=torch.zeros((batch,desc_dim,max_len))
            image_size = np.repeat(image_size[np.newaxis, :], batch, axis=0)
            keypoints=torch.zeros((batch,max_len,2))
            scores=torch.zeros((batch,max_len))
            for i,feat in enumerate(feats1):
                descriptors_=feat['descriptors'].__array__()
                dim=descriptors_.shape[1]
                descriptors[i,:,:dim]=torch.tensor(descriptors_)
                keypoints[i,:dim,:]=torch.tensor(feat['keypoints'].__array__())
                scores[i,:dim]=torch.tensor(feat['scores'].__array__())
            pred['descriptors1']=descriptors.to(self.device)
            pred['image_size1']=torch.tensor(image_size).to(self.device)
            pred['keypoints1']=keypoints.to(self.device)
            pred['keypoint_scores1']=scores.to(self.device)


            pred1=self.local_feature_matcher(pred)
            matches = pred1['matches0'].detach().cpu().short().numpy()

            registered_feature = [self.registered_feature[i] for i in topk]
            for ind,match in enumerate(matches):
                landmark_position=registered_feature[ind]
                index_list=set(landmark_position[0])
                feature_dict = {key: value for key, value in zip(landmark_position[0], landmark_position[1])}
                pts0 = np.empty((len(match), 2))
                pts1 = np.empty((len(match), 2))
                lms = np.empty((len(match),3))
                counter=0
                feat0,feat1=pred['keypoints0'][ind].detach().cpu().numpy(),pred['keypoints1'][ind].detach().cpu().numpy()
                for n,m in enumerate(match):
                    if m != -1 and m in index_list:
                        pts0[counter] = feat0[n]
                        pts1[counter] = feat1[m]
                        lms[counter] = feature_dict[m]
                        counter += 1
                if counter>parent.thre:
                    pts0_list.append(pts0[:counter])
                    pts1_list.append(pts1[:counter])
                    lms_list.append(lms[:counter])
                    parent.retrived_image_index.append(topk[ind])
            del pred,pred1
        return pts0_list,pts1_list,lms_list,max_dim

        

    def lightglue(self, i, feats0):
        feats1 = self.hfile_local[self.db_name[i]]

        # Batch data transfer to GPU
        pred = {
            **{k+'0': torch.tensor(np.array(v)).unsqueeze(0) for k, v in feats0.items()},
            **{k+'1': torch.tensor(np.array(v)).unsqueeze(0) for k, v in feats1.items()},
        }
        pred = {k: v.to(self.device) for k, v in pred.items()}
        pred['keypoint_scores0'] = pred.pop('scores0')
        pred['keypoint_scores1'] = pred.pop('scores1')

        with torch.inference_mode():
            pred = self.local_feature_matcher(pred)

        matches = pred['matches0'][0].detach().cpu().short().numpy()

        registered_feature = self.registered_feature[i]
        index_list = set(registered_feature[0])
        feature_dict = {key: value for key, value in zip(registered_feature[0], registered_feature[1])}

        # Reuse memory allocation
        pts0, pts1, lms = self.match_filter(matches, feats0, feats1, feature_dict, index_list)

        return [pts0, pts1, lms]
    
        
    def superglue(self, i, feats0):
        feats1 = self.hfile_local[self.db_name[i]]
        data = {}
        for k in feats0.keys():
            data[k + '0'] = feats0[k]
        for k in feats0.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(self.device)
                for k, v in data.items()}
        data['image0'] = torch.empty((1, 1,) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,) + tuple(feats1['image_size'])[::-1])

        with torch.inference_mode():
            pred = self.local_feature_matcher(data)
        matches = pred['matches0'][0].detach().cpu().short().numpy()
        registered_feature = self.registered_feature[i]
        index_list = set(registered_feature[0])
        feature_dict = {key: value for key, value in zip(registered_feature[0], registered_feature[1])}
        pts0 = np.empty((len(matches), 2))
        pts1 = np.empty((len(matches), 2))
        lms = np.empty((len(matches),3))
        counter = 0
        for n, m in enumerate(matches):
            if m != -1 and m in index_list:
                pts0[counter] = feats0['keypoints'][n]
                pts1[counter] = feats1['keypoints'][m]
                lms[counter] = feature_dict[m]
                counter += 1
        pts0 = pts0[:counter]
        pts1 = pts1[:counter]
        lms = lms[:counter]
        return [pts0,pts1,lms]