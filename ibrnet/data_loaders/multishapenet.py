#from srt.utils.nerf import get_extrinsic, transform_points
#from srt.utils.common import get_rank, get_world_size
from ibrnet.data_loaders.nerf import get_extrinsic, transform_points, get_extrinsic_inv
from ibrnet.data_loaders.common import get_rank, get_world_size

from torch.utils.data import get_worker_info, IterableDataset
from torch.utils.data import Dataset
import numpy as np
import ipdb
import torch



class MultishapenetDataset(IterableDataset):
    def __init__(self, args, mode, points_per_item=8192, max_len=None,
                 full_scale=False, scenes=None):
        super(MultishapenetDataset).__init__()
        self.num_target_pixels = points_per_item
        self.path = "/home/ubuntu/IBRNet/data/msn"
        self.mode = mode
        self.full_scale = full_scale

        self.render_kwargs = {
            'min_dist': 0.,
            'max_dist': 20.}

        import sunds  # Import here, so that only this dataset depends on Tensorflow
        builder = sunds.builder('multi_shapenet', data_dir=self.path)

        

        self.tf_dataset = builder.as_dataset(
            split=self.mode, 
            task=sunds.tasks.Nerf(yield_mode='stacked'),
        )

        self.num_items = 1000000 if mode == 'train' else 10000
        if max_len is not None:
            self.num_items = min(max_len, self.num_items)

        self.tf_dataset = self.tf_dataset.take(self.num_items)
        if self.mode == "train":
            self.tf_dataset = self.tf_dataset.shuffle(1024)
        self.tf_iterator = self.tf_dataset.as_numpy_iterator()
        
        #ipdb.set_trace()
        #ipdb.set_trace()

    def __len__(self):
        return len(self.tf_dataset)

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()

        dataset = self.tf_dataset

        if world_size > 1:
            num_shardable_items = (self.num_items // world_size) * world_size
            if num_shardable_items != self.num_items:
                print(f'MSN: Using {num_shardable_items} scenes to {self.mode} instead of {self.num_items} to be able to evenly shard to {world_size} processes.')
                dataset = dataset.take(num_shardable_items)
            dataset = dataset.shard(num_shards=world_size, index=rank)

        if self.mode == 'train':
            dataset = dataset.shuffle(1024)
        tf_iterator = dataset.as_numpy_iterator()

        for data in tf_iterator:
            yield self.prep_item(data)

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()

        dataset = self.tf_dataset
        ipdb.set_trac()

        if world_size > 1:
            num_shardable_items = (self.num_items // world_size) * world_size
            if num_shardable_items != self.num_items:
                print(f'MSN: Using {num_shardable_items} scenes to {self.mode} instead of {self.num_items} to be able to evenly shard to {world_size} processes.')
                dataset = dataset.take(num_shardable_items)
            dataset = dataset.shard(num_shards=world_size, index=rank)

        if self.mode == 'train':
            dataset = dataset.shuffle(1024)
        tf_iterator = dataset.as_numpy_iterator()
        #ipdb.set_trace()
        
        for data in tf_iterator:
            ipdb.set_trac()
            yield self.prep_item(data)

    def prep_item(self, data):

        input_views = np.random.choice(np.arange(10), size=5, replace=False)
        target_views = np.array(list(set(range(10)) - set(input_views)))
        
        target_view = np.random.randint(5)
        target_view = target_views[target_view]

        data['color_image'] = data['color_image'].astype(np.float32) / 255.

        input_images = np.transpose(data['color_image'][input_views], (0, 3, 1, 2))
        input_rays = data['ray_directions'][input_views]
        input_camera_pos = data['ray_origins'][input_views][:, 0, 0]
        
        
        source_camera_pos_list = []
        for i in range(5):
            source_camera_pos_list.append(get_extrinsic(input_camera_pos[i], input_rays[i]).astype(np.float32))
        
        source_camera_pos = np.concatenate(source_camera_pos_list,axis=0)
        #if self.canonical:
            #canonical_extrinsic = get_extrinsic(input_camera_pos[0], input_rays[0]).astype(np.float32)
            #input_rays = transform_points(input_rays, canonical_extrinsic, translate=False)
            #input_camera_pos = transform_points(input_camera_pos, canonical_extrinsic)
        
        
        target_pixels = data['color_image'][target_view]
        #target_pixels = np.reshape(data['color_image'][target_views], (-1, 3))
        target_rays = data['ray_directions'][target_view]
        target_camera_pos = data['ray_origins'][target_view][0, 0]
        target_camera_pos = get_extrinsic(target_camera_pos, target_rays).astype(np.float32)
        depth_range = [0., 20.]
        #ipdb.set_trace()



        #num_pixels = target_pixels.shape[0]

        #if not self.full_scale:
            #sampled_idxs = np.random.choice(np.arange(num_pixels),
                                            #size=(self.num_target_pixels,),
                                            #replace=False)

            #target_pixels = target_pixels[sampled_idxs]
            #target_rays = target_rays[sampled_idxs]
            #target_camera_pos = target_camera_pos[sampled_idxs]

        #if self.canonical:
            #target_rays = transform_points(target_rays, canonical_extrinsic, translate=False)
            #target_camera_pos = transform_points(target_camera_pos, canonical_extrinsic)

        #sceneid = int(data['scene_name'][6:])

        result = {
            'rgb_path':        torch.from_numpy(input_images).permute(0,2,3,1),         # [k, h, w,3]
            #'src_cameras':     torch.from_numpy(source_camera_pos),    # [k, 3]
            #'input_rays':                                 input_rays,           # [k, h, w, 3]
           # 'rgb':        torch.from_numpy(target_pixels),        # [h,w, 3]
            #'target_camera_pos':    target_camera_pos,    # [p, 3]
            #'camera':          torch.from_numpy(target_rays),          # [p, 3]
            #'sceneid':              #sceneid,              # int
            #'depth_range': depth_range,
        }
          # [3, 4] (optional)
        return result


    
    # def __getitem__(self,idx):
    #     #data = self.tf_dataset[idx]
    #     #data = next(iter(self.tf_dataset.skip(idx).take(1)))
    #     data= next(self.tf_iterator)
    #     #ipdb.set_trace()
    #     #data['scene_name'] = bytes.decode(data['scene_name'].numpy())
    #     data['scene_name'] = bytes.decode(data['scene_name'])
        
    #     input_views = np.random.choice(np.arange(10), size=5, replace=False)
    #     target_views = np.array(list(set(range(10)) - set(input_views)))

        
    #     target_view = np.random.randint(5)
    #     target_view = target_views[target_view]

    #     data['color_image'] = data['color_image'].astype(np.float32) / 255.

    #     input_images = np.transpose(data['color_image'][input_views], (0, 3, 1, 2))
    #     input_rays = data['ray_directions'][input_views]
    #     input_camera_pos = data['ray_origins'][input_views][:, 0, 0]
        
    #     source_camera_pos_list = []
    #     for i in range(5):
    #         source_camera_pos_list.append(get_extrinsic(input_camera_pos[i], input_rays[i]).astype(np.float32))
        

    #     #if self.canonical:
    #         #canonical_extrinsic = get_extrinsic(input_camera_pos[0], input_rays[0]).astype(np.float32)
    #         #input_rays = transform_points(input_rays, canonical_extrinsic, translate=False)
    #         #input_camera_pos = transform_points(input_camera_pos, canonical_extrinsic)
        
        
    #     target_pixels = data['color_image'][target_view]
    #     #target_pixels = np.reshape(data['color_image'][target_views], (-1, 3))
    #     target_rays = data['ray_directions'][target_view]
    #     target_camera_pos = data['ray_origins'][target_view][0, 0]
    #     target_camera_pos = get_extrinsic(target_camera_pos, target_rays).astype(np.float32)
    #     depth_range = [0., 20.]



    #     #num_pixels = target_pixels.shape[0]

    #     #if not self.full_scale:
    #         #sampled_idxs = np.random.choice(np.arange(num_pixels),
    #                                         #size=(self.num_target_pixels,),
    #                                         #replace=False)

    #         #target_pixels = target_pixels[sampled_idxs]
    #         #target_rays = target_rays[sampled_idxs]
    #         #target_camera_pos = target_camera_pos[sampled_idxs]

    #     #if self.canonical:
    #         #target_rays = transform_points(target_rays, canonical_extrinsic, translate=False)
    #         #target_camera_pos = transform_points(target_camera_pos, canonical_extrinsic)

    #     #sceneid = int(data['scene_name'][6:])

    #     #ipdb.set_trace()

    #     result = {
    #         'rgb_path':        torch.from_numpy(input_images.permute(0,2,3,1)),         # [k, h, w,3]
    #         'src_cameras':     torch.from_numpy(source_camera_pos_list),    # [k, 3]
    #         #'input_rays':                                 input_rays,           # [k, h, w, 3]
    #         'rgb':        torch.from_numpy(target_pixels),        # [h,w, 3]
    #         #'target_camera_pos':    target_camera_pos,    # [p, 3]
    #         'camera':          torch.from_numpy(target_rays),          # [p, 3]
    #         #'sceneid':              #sceneid,              # int
    #         'depth_range': torch.from_numpy(depth_range),
    #     }
    #       # [3, 4] (optional)
    #     return result

        

    # def __iter__(self):
    #     rank = get_rank()
    #     world_size = get_world_size()

    #     dataset = self.tf_dataset

    #     if world_size > 1:
    #         num_shardable_items = (self.num_items // world_size) * world_size
    #         if num_shardable_items != self.num_items:
    #             print(f'MSN: Using {num_shardable_items} scenes to {self.mode} instead of {self.num_items} to be able to evenly shard to {world_size} processes.')
    #             dataset = dataset.take(num_shardable_items)
    #         dataset = dataset.shard(num_shards=world_size, index=rank)

    #     if self.mode == 'train':
    #         dataset = dataset.shuffle(1024)
    #     tf_iterator = dataset.as_numpy_iterator()

    #     for data in tf_iterator:
    #         yield self.prep_item(data)

    # def prep_item(self, data):
    #     input_views = np.random.choice(np.arange(10), size=5, replace=False)
    #     target_views = np.array(list(set(range(10)) - set(input_views)))
        
    #     target_view = np.random.randint(5)
    #     target_view = target_views[target_view]

    #     data['color_image'] = data['color_image'].astype(np.float32) / 255.

    #     input_images = np.transpose(data['color_image'][input_views], (0, 3, 1, 2))
    #     input_rays = data['ray_directions'][input_views]
    #     input_camera_pos = data['ray_origins'][input_views][:, 0, 0]
        
    #     source_camera_pos_list = []
    #     for i in range(5):
    #         source_camera_pos_list.append(get_extrinsic(input_camera_pos[i], input_rays[i]).astype(np.float32))
        

    #     #if self.canonical:
    #         #canonical_extrinsic = get_extrinsic(input_camera_pos[0], input_rays[0]).astype(np.float32)
    #         #input_rays = transform_points(input_rays, canonical_extrinsic, translate=False)
    #         #input_camera_pos = transform_points(input_camera_pos, canonical_extrinsic)
        
        
    #     target_pixels = data['color_image'][target_view]
    #     #target_pixels = np.reshape(data['color_image'][target_views], (-1, 3))
    #     target_rays = np.reshape(data['ray_directions'][target_view], (-1, 3))
    #     target_camera_pos = data['ray_origins'][target_view][:, 0, 0]
    #     target_camera_pos = get_extrinsic(target_camera_pos, target_rays).astype(np.float32)
    #     depth_range = [0., 20.]



    #     #num_pixels = target_pixels.shape[0]

    #     #if not self.full_scale:
    #         #sampled_idxs = np.random.choice(np.arange(num_pixels),
    #                                         #size=(self.num_target_pixels,),
    #                                         #replace=False)

    #         #target_pixels = target_pixels[sampled_idxs]
    #         #target_rays = target_rays[sampled_idxs]
    #         #target_camera_pos = target_camera_pos[sampled_idxs]

    #     #if self.canonical:
    #         #target_rays = transform_points(target_rays, canonical_extrinsic, translate=False)
    #         #target_camera_pos = transform_points(target_camera_pos, canonical_extrinsic)

    #     #sceneid = int(data['scene_name'][6:])

    #     result = {
    #         'rgb_path':        input_images.permute(0,2,3,1),         # [k, h, w,3]
    #         'src_cameras':     source_camera_pos_list,    # [k, 3]
    #         #'input_rays':                                 input_rays,           # [k, h, w, 3]
    #         'rgb':        target_pixels,        # [h,w, 3]
    #         #'target_camera_pos':    target_camera_pos,    # [p, 3]
    #         'camera':          target_rays,          # [p, 3]
    #         #'sceneid':              #sceneid,              # int
    #         'depth_range': depth_range,
    #     }
    #       # [3, 4] (optional)
    #     return result

    def skip(self, n):
        """
        Skip the first n scenes
        """
        self.tf_dataset = self.tf_dataset.skip(n)




