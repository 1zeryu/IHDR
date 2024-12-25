from pit.config import cfg
import torch
import numpy as np
import torchvision.transforms as T
import random
from pit.datasets import make_dataloader
from pit.model import make_model
from pit.utils.metrics import euclidean_distance
from PIL import Image
import os
from tqdm import tqdm
from decord import VideoReader


class PiTExtractor:
    def __init__(self, config_file='pit/configs/iLIDS-VID/pit.yml', device=0, model_path='/mnt/pfs/users/yukun.zhou/codes/github/pit/logs/ilids_PiT/1/transformer_60.pth'):
        if config_file != "":
            cfg.merge_from_file(config_file)
            cfg.freeze()

        if cfg.MODEL.DIST_TRAIN:
            torch.cuda.set_device(device)

        self.val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
        train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
        self.model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        self.model.load_param(model_path)

        if isinstance(device, int):
            self.model.to("cuda:{}".format(device))
            self.device = "cuda:{}".format(device)
        else:
            self.model.to(device)
            self.device = device

        self.model.eval()

        self.database = None
        if self.database is None:
            self.get_database()

    def prepare_input(self, input):
        def get_raw(input):
            if isinstance(input, str):
                with Image.open(input) as im:
                    img = im.convert('RGB')
                img = [self.val_transforms(img)]
                return img
            elif isinstance(input, list):
                input_list = []
                for img in input:
                    raw_list = get_raw(img)
                    input_list.append(raw_list[0])
                return input_list
            elif isinstance(input, Image.Image):
                img = input.convert('RGB') 
                return [self.val_transforms(img)]
            else:
                raise ValueError("Input type not supported")
                
        input = get_raw(input)
        input = torch.stack(input, 0).unsqueeze(0).to(self.device)
        return input
    
    def encode(self, img, cam_label=[0]):
        input = self.prepare_input(img)
        feat = self.model(input, cam_label=cam_label).detach().cpu()
        return feat
    
    def encode_batch(self, img_list, cam_label=[0]):
        img_list = [self.prepare_input(img) for img in img_list]
        feat_list = []
        for img in tqdm(img_list, desc="Extracting features"):
            feat = self.model(img, cam_label=cam_label).detach().cpu()
            feat_list.append(feat)
        return torch.cat(feat_list, dim=0)

    
    def get_database(self,):
        database_feat_path = 'database_feat.pth'
        if os.path.exists(database_feat_path):
            self.database, self.database_file_list = torch.load(database_feat_path)
        else:
            database_file_list = []
            img_dir_root = '/mnt/pfs/users/yukun.zhou/codes/github/pit/i-LIDS-VID/images/cam1'
            img_dir_list = os.listdir(img_dir_root)
            for img_dir in img_dir_list:
                img_list = os.listdir(os.path.join(img_dir_root, img_dir))
                for img in img_list:
                    database_file_list.append(os.path.join(img_dir_root, img_dir, img))
            print("database length: ", len(database_file_list))
            database_feat = extractor.encode_batch(database_file_list)
            torch.save([database_feat, database_file_list], database_feat_path)
            self.database = database_feat
            self.database_file_list = database_file_list

        self.database = torch.nn.functional.normalize(self.database, dim=1, p=2)

    def rank_query(self, query_feat, look_num=10):
        query_feat = torch.nn.functional.normalize(query_feat, dim=1, p=2)
        dist = euclidean_distance(query_feat, self.database).reshape(-1)
        rank = np.argsort(dist)
        rank = rank[:look_num] if look_num != -1 and look_num < len(rank) else rank
        return rank, dist[rank], [self.database_file_list[i] for i in rank], [self.database[i] for i in rank]
    
    def query(self, img, look_num=10):
        query_feat = self.encode(img)
        return self.rank_query(query_feat, look_num=look_num)
    
    def query_video(self, video, look_num=10):
        # video to list of images
        video = VideoReader(video)
        video = [img.asnumpy() for img in video]
        video = [Image.fromarray(img) for img in video]
        query_feat = self.encode(video)
        return self.rank_query(query_feat, look_num=look_num)
    

if __name__ == "__main__":
    extractor = PiTExtractor(config_file='pit/configs/iLIDS-VID/pit.yml', 
                             device='cuda:1', model_path='/mnt/pfs/users/yukun.zhou/codes/github/pit/logs/ilids_PiT/1/transformer_60.pth')

    # database_file_list = []
    # img_dir_root = '/mnt/pfs/users/yukun.zhou/codes/github/pit/i-LIDS-VID/images/cam1'
    # img_dir_list = os.listdir(img_dir_root)
    # for img_dir in img_dir_list:
    #     img_list = os.listdir(os.path.join(img_dir_root, img_dir))
    #     for img in img_list:
    #         database_file_list.append(os.path.join(img_dir_root, img_dir, img))

    # database_feat = extractor.encode(database_file_list)

    extractor.get_database()

    query_file_list = ['/mnt/pfs/users/yukun.zhou/codes/github/pit/i-LIDS-VID/images/cam2/person001/cam2_person001.png']
    query_feat = extractor.encode(query_file_list)
    rank, rank_dist, rank_file, rank_feat = extractor.rank_query(query_feat)





    
    


                

        