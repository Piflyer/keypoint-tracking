# Predict from a model and save results
from ultralytics import YOLO
import glob
import numpy as np
from PIL import Image
import json

# Save format:
# For CAST: xyz keypoints (mm) for each image
# For plotting/visualization: pixel keypoints, CAST pose, rgb image path
# there should be a different file for every mug and every scene

def pixel_to_world(kpts_pixel, cam_K, depth):
    kpts_px = np.hstack([kpts_pixel, np.ones([kpts_pixel.shape[0], 1])])
    kpts_world = depth[:, np.newaxis] * (np.linalg.inv(cam_K) @ kpts_px.T).T
    return kpts_world

def main():
    # get cam_K (hardcoded because... NOCS)
    cam_K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    # Load a model
    model = YOLO(model_savefile)  # load a custom model
    model2 = YOLO(model_savefile2)

    save_results = dict()
    failure_count = 0
    objimg_count = 0
    # loop through scenes
    for scene in expected_count:
        # if scene != "scene_1":
        #     break
        if expected_count[scene] == 0:
            # why run the model?
            continue

        # create save for scene
        save_results[scene] = dict()

        rgb_paths = sorted(glob.glob(f"{dataset_parent_path}/{scene}/*_color.png"))
        depth_paths = sorted(glob.glob(f"{dataset_parent_path}/{scene}/*_depth.png"))
        meta_paths = sorted(glob.glob(f"{dataset_parent_path}/{scene}/*_meta.txt"))
        gt_paths = sorted(glob.glob(f"{dataset_gt_path}_{scene}_*.pkl"))
        scene_objs = []
        
        # loop through images
        for idx, _ in enumerate(rgb_paths):
            objimg_count += expected_count[scene]
            # get gts
            # 2) load meta, 1) load gt pickle, 3) pick gt from
            gts_all = np.load(gt_paths[idx], allow_pickle=True)['gt_RTs']
            gts = []
            with open(meta_paths[idx]) as f:
                for line in f:
                    meta = line.split() # number, class id, name
                    if int(meta[1]) == class_id:
                        # first time: update scene_objs
                        if len(scene_objs) < expected_count[scene]:
                            scene_objs.append(meta[2])
                            save_results[scene][scene_objs[-1]] = []
                        # save gts
                        gt_idx = int(meta[0])-1
                        if gt_idx < gts_all.shape[0]:
                            RT_gt = gts_all[gt_idx,:,:]
                            # NOCS ROTATION MATRIX IS NOT NORMALIZED!
                            # BECAUSE SOMEHOW IN 2019 THAT WAS OKAY
                            RT_gt[:,0] = RT_gt[:,0] / np.linalg.norm(RT_gt[:,0])
                            RT_gt[:,1] = RT_gt[:,1] / np.linalg.norm(RT_gt[:,1])
                            RT_gt[:,2] = RT_gt[:,2] / np.linalg.norm(RT_gt[:,2])
                            gts.insert(scene_objs.index(meta[2]), RT_gt)
                        else:
                            # no gt in list
                            gts.insert(scene_objs.index(meta[2]), np.eye(4))


            # load depth image
            depths = np.array(Image.open(depth_paths[idx]), dtype=float) / 1000. # m
            depths = depths.T

            # run prediction
            results = model(rgb_paths[idx])
            result = results[0]

            results2 = model2(rgb_paths[idx])
            result2 = results2[0]
            result2_num = 0
            result_num = 0
            if result2.keypoints.has_visible:
                result2_num = np.sum(result2.keypoints.xy[0,:,0].cpu().numpy() > 0)
            if result.keypoints.has_visible:
                result_num = np.sum(result.keypoints.xy[0,:,0].cpu().numpy() > 0)
            if result2_num > result_num:
                result = result2
            # if scene == 'scene_5':
            #     result.show()
            #     input()

            # if no detections, report 0
            if not result.keypoints.has_visible:
                num_keypoints = int(result.keypoints.shape[2] / 3)
                for i in range(expected_count[scene]):
                    failure_count += 1
                    res_cur = dict()
                    res_cur['est_pixel_keypoints'] = np.zeros([num_keypoints,2]).tolist()
                    res_cur['est_world_keypoints'] = np.zeros([num_keypoints,3]).tolist()
                    res_cur['rgb_image_filename']  = rgb_paths[idx]
                    res_cur['gt_pose'] = gts[i].tolist()

                    # if (scene == "scene_1"):
                    #     gt_kpt_path = "/home/lorenzo/research/tracking/datasets/bop/nocs/models/mug_daniel_norm.csv"
                    #     gt_kpt_canonical = np.genfromtxt(gt_kpt_path, delimiter=",") / 1000. # [m]
                    #     gt_kpts_cam = gts[0] @ np.array([[1.,0,0,0], [0,0,1,0],[0,-1,0,0],[0,0,0,1]]) @ np.hstack([gt_kpt_canonical, np.ones([gt_kpt_canonical.shape[0],1])]).T
                    #     res_cur['gt_world_keypoints'] = gt_kpts_cam[:3,:].T.tolist()
                    #     # TODO: gt world keypoints with occlusion!

                    if (scene == "scene_1"):
                        res_cur['est_world_keypoints_spherical'] = np.zeros([num_keypoints*9,3]).tolist()

                    save_results[scene][scene_objs[i]].append(res_cur)
                continue

            # for 1 expected detection, just save
            if expected_count[scene] == 1:
                res_cur = dict()
                # use most confident detections only (result.boxes.conf)
                # seems to be sorted by confidence, just pick first
                kpts_pixel = result.keypoints.xy[0,:,:].cpu().numpy()
                res_cur['est_pixel_keypoints'] = kpts_pixel.tolist()
                kpts_pixel_int = np.rint(kpts_pixel).astype(int)
                # clamp
                kpts_pixel_int[:,0] = np.clip(kpts_pixel_int[:,0], 0, depths.shape[0]-1)
                kpts_pixel_int[:,1] = np.clip(kpts_pixel_int[:,1], 0, depths.shape[1]-1)
                depth = depths[kpts_pixel_int[:,0], kpts_pixel_int[:,1]]
                res_cur['est_world_keypoints'] = pixel_to_world(kpts_pixel, cam_K, depth).tolist()
                res_cur['rgb_image_filename']  = rgb_paths[idx]
                res_cur['gt_pose'] = gts[0].tolist()

                # gt keypoints for scene_1 only
                # if (scene == "scene_1"):
                #     gt_kpt_path = "/home/lorenzo/research/tracking/datasets/bop/nocs/models/mug_daniel_norm.csv"
                #     gt_kpt_canonical = np.genfromtxt(gt_kpt_path, delimiter=",") / 1000. # [m]
                #     gt_kpts_cam = gts[0] @ np.array([[1.,0,0,0], [0,0,1,0],[0,-1,0,0],[0,0,0,1]]) @ np.hstack([gt_kpt_canonical, np.ones([gt_kpt_canonical.shape[0],1])]).T
                #     res_cur['gt_world_keypoints'] = gt_kpts_cam[:3,:].T.tolist()
                #     # no occlusion!

                # spherical keypoints for scene_1 only
                if (scene == "scene_1"):
                    # search in 1 pixel radius
                    mod = np.array([[0,0],[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1]])
                    kpts_pixel_spherical = np.repeat(kpts_pixel_int.T, repeats=mod.shape[0], axis=1)
                    kpts_pixel_spherical += np.tile(mod,(kpts_pixel_int.shape[0],1)).T
                    # clamp
                    kpts_pixel_spherical[0,:] = np.clip(kpts_pixel_spherical[0,:], 0, depths.shape[0]-1)
                    kpts_pixel_spherical[1,:] = np.clip(kpts_pixel_spherical[1,:], 0, depths.shape[1]-1)
                    # compute pixel depth
                    kpts_spherical_depths = depths[kpts_pixel_spherical[0,:], kpts_pixel_spherical[1,:]]
                    res_cur['est_world_keypoints_spherical'] = pixel_to_world(kpts_pixel_spherical.T, cam_K, kpts_spherical_depths).tolist()

                    # Below: nice way of minimizing duplicates with spherical, but will require non-negligible refactor of code!
                    # # compare depths for each repeated keypoint
                    # depths_reshaped = kpts_spherical_depths.copy().reshape([-1,mod.shape[0]])
                    # depths_reshaped -= depths_reshaped[:,0].reshape([-1,1])
                    # unique_idxs = np.array([])
                    # spherical_lib_count = []
                    # for row in range(depths_reshaped.shape[0]):
                    #     _, unique_idx = np.unique(depths_reshaped[row,:].round(decimals=2), return_index=True)
                    #     spherical_lib_count.append(len(unique_idx))
                    #     unique_idxs = np.append(unique_idxs, unique_idx + row*mod.shape[0])
                    # kpts_world = pixel_to_world(kpts_pixel_spherical.T, cam_K, kpts_spherical_depths)

                    # res_cur['est_world_keypoints_spherical'] = kpts_world[unique_idxs.astype(int),:].tolist()
                    # res_cur['spherical_lib_count'] = spherical_lib_count
                    # # to use: duplicate each shape in shape lib according to spherical lib count

                save_results[scene][scene_objs[0]].append(res_cur)

            elif expected_count[scene] == 2:
                detection_count = result.keypoints.shape[0]
                # project bb center to world and compare with gt pose
                bbcenters = result.boxes.xywh.cpu().numpy()[:,:2]
                for i in range(expected_count[scene]):
                    if i >= detection_count:
                        failure_count += 1
                        break
                    center_px = bbcenters[i,:]
                    center_px_int = np.rint(center_px).astype(int) 
                    depth = depths[center_px_int[0], center_px_int[1]]
                    center_px = np.hstack([center_px, 1])
                    center_xyz = depth * (np.linalg.inv(cam_K) @ center_px.T)
                    # compare center xyz with gt translations
                    gt_trans = np.stack(gts)[:,:3,3]
                    dist = np.linalg.norm(gt_trans - center_xyz,axis=1)
                    save_idx = np.argmin(dist)

                    # save
                    res_cur = dict()
                    kpts_pixel = result.keypoints.xy[i,:,:].cpu().numpy()
                    res_cur['est_pixel_keypoints'] = kpts_pixel.tolist()
                    kpts_pixel_int = np.rint(kpts_pixel).astype(int)
                    # clamp
                    kpts_pixel_int[:,0] = np.clip(kpts_pixel_int[:,0], 0, depths.shape[0]-1)
                    kpts_pixel_int[:,1] = np.clip(kpts_pixel_int[:,1], 0, depths.shape[1]-1)
                    depth = depths[kpts_pixel_int[:,0], kpts_pixel_int[:,1]]
                    res_cur['est_world_keypoints'] = pixel_to_world(kpts_pixel, cam_K, depth).tolist()
                    res_cur['rgb_image_filename']  = rgb_paths[idx]
                    res_cur['gt_pose'] = gts[save_idx].tolist()
                    save_results[scene][scene_objs[i]].append(res_cur)


    print(f"{failure_count}/{objimg_count} failed!")
    print("Saving to JSON...")

    # save all in JSONs for processing
    for scene in save_results:
        scene_results = save_results[scene]
        for obj in scene_results:
            obj_results = scene_results[obj]
            data = json.dumps(obj_results)
            filename = f"{save_path}/{scene}-{obj}.json"
            with open(filename, 'w') as f:
                f.write(data)


if __name__ == '__main__':
    # model_savefile = "runs/pose/train9/weights/epoch50.pt"
    # model_savefile2 = "runs/pose/train9/weights/best.pt"
    # model_savefile = "runs/pose/train20/weights/epoch50.pt"
    # model_savefile2 = "runs/pose/train20/weights/best.pt"
    model_savefile = "runs/pose/train17/weights/epoch50.pt"
    model_savefile2 = "runs/pose/train17/weights/best.pt"
    dataset_parent_path = "NOCS/real_test"
    dataset_gt_path = "NOCS/gts/real_test/results_real_test"
    save_path = "mugs/NOCS/camera3_2"

    # class_id = 6 # = mugs (see NOCS meta)
    # expected_count = {"scene_1": 1, "scene_2": 1, "scene_3": 1, "scene_4": 1, "scene_5": 0, "scene_6": 2} # if > detected, reduce via confidence

    class_id = 3 # = camera
    expected_count = {"scene_1": 1, "scene_2": 1, "scene_3": 1, "scene_4": 1, "scene_5": 2, "scene_6": 0} # if > detected, reduce via confidence

    main()