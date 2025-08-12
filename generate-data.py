import blenderproc as bproc
import argparse
import os
import numpy as np
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()


parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2000, help="How many scenes with 25 images each to generate")
args = parser.parse_args()

bproc.init()

# load bop objects into the scene
# target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True)

# load distractor bop objects
# tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
# hb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'hb'), mm2m = True)
# tyol_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tyol'), mm2m = True)
ycb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True, obj_ids=[2,3,4,5,8,9,10,15,16,17,18,19,20,21])
ycb_dist_lmo_obs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'lmo'), mm2m = True, obj_ids=[1,5,6,8,9,10,11,12])


# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'))

fork_ids = [1,2,3,4,5,6,7,8,9,10,12,13,14,15]
target_forks = []
for j in fork_ids:
    target_forks.append(bproc.loader.load_obj(filepath=os.path.join(args.bop_parent_path, 'fork', 'models', f'obj_{j:06d}.glb'), object_model_unit='m')[-1])

for obj in (target_forks):
    # obj.set_shading_mode('auto')
    obj.set_cp("category_id", 1)
    obj.set_cp("bop_dataset_name", "timn")
    obj.hide(True)

# set shading and hide objects
for obj in (ycb_dist_bop_objs + ycb_dist_lmo_obs):
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[3, 3, 1]),
               bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, -3, 3], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 3, 3], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[3, 0, 3], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[-3, 0, 3], rotation=[0, 1.570796, 0])]
# for plane in room_planes:
#     plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(300)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

#validate the objects
valid_forks = []
for obj in target_forks:
    if not obj.get_mesh():
        print(f"Object {obj.get_name()} has no vertices.")
        #remove object from scene
        obj.delete()
        continue
    if len(obj.get_mesh().vertices) < 100:
        print(f"Object {obj.get_name()} has less than 100 vertices.")
        #remove object from scene
        obj.delete()
        continue
    valid_forks.append(obj)

target_forks = valid_forks

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

for i in range(args.num_scenes):

    qualified = False

    while not qualified:
        # Ensure ALL objects are hidden and rigidbody is disabled
        for obj in (target_forks + ycb_dist_bop_objs + ycb_dist_lmo_obs):
            obj.hide(True)
            obj.disable_rigidbody()
            obj.clear_parent()  # Clear any parent relationships that might cause issues

        # Sample bop objects for a scene
        sampled_target_bop_objs = list(np.random.choice(target_forks, size=1, replace=False))
        sampled_distractor_bop_objs = list(np.random.choice(ycb_dist_bop_objs, size=6, replace=False))
        sampled_distractor_bop_objs += list(np.random.choice(ycb_dist_lmo_obs, size=4, replace=False))
        
        print(f"Sampled target fork: {sampled_target_bop_objs[0].get_name()}")
        print(f"Total objects to be placed: {len(sampled_target_bop_objs + sampled_distractor_bop_objs)}")

        # Randomize materials and set physics
        for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):        
            # mat = obj.get_materials()[0]
            # if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            #     grey_col = np.random.uniform(0.1, 0.9)   
            #     mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
            # mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            # mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
            try:
                obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            except Exception as e:
                print(f"Error enabling rigidbody for {obj.get_name()}: {e}")

            obj.hide(False)
            print(f"Object {obj.get_name()} with category_id {obj.get_cp('category_id')} and bop_dataset_name {obj.get_cp('bop_dataset_name')} is sampled.")
        
        # Double-check that no unintended forks are visible
        all_visible_forks = [obj for obj in target_forks if not obj.is_hidden()]
        print(f"All visible forks after setup: {[obj.get_name() for obj in all_visible_forks]}")
        if len(all_visible_forks) != 1:
            print(f"WARNING: Expected 1 visible fork, but found {len(all_visible_forks)}")
            # Hide any unintended visible forks
            for obj in all_visible_forks:
                if obj not in sampled_target_bop_objs:
                    print(f"Hiding unintended fork: {obj.get_name()}")
                    obj.hide(True)
        
        # Sample two light sources
        light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                                elevation_min = 5, elevation_max = 89)
        light_point.set_location(location)

        # sample CC Texture and assign to room planes
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)


        # # Sample object poses and check collisions 
        # bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs + sampled_distractor_bop_objs,
        #                         sample_pose_func = sample_pose_func, 
        #                         max_tries = 1000)
                
        # # Physics Positioning
        # bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
        #                                                 max_simulation_time=10,
        #                                                 check_object_interval=1,
        #                                                 substeps_per_frame = 20,
        #                                                 solver_iters=25)
        
        # Sample object poses and check collisions
        bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs + sampled_distractor_bop_objs,
                                sample_pose_func = sample_pose_func,
                                objects_to_check_collisions = sampled_target_bop_objs + sampled_distractor_bop_objs, # added as test
                                max_tries = 100)

        # Physics Positioning
        def sample_initial_pose(obj: bproc.types.MeshObject):
            obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                        min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
            obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

        # Sample objects on the given surface
        placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
                                                            surface=room_planes[0],
                                                            sample_pose_func=sample_initial_pose,
                                                            min_distance=0.001,
                                                            max_distance=0.5)

        #print final count of placed_objects
        print(f"Final count of placed_objects: {len(placed_objects)}")
        #check if multiple forks are visible
        visible_forks = sum(obj.is_hidden() == False for obj in placed_objects if obj.get_cp("category_id") == 1)
        print(f"Visible forks: {visible_forks}")
        
        # Final verification - count all visible forks in the entire scene, not just placed objects
        all_scene_visible_forks = sum(obj.is_hidden() == False for obj in target_forks if obj.get_cp("category_id") == 1)
        print(f"All visible forks in scene: {all_scene_visible_forks}")
        
        if len(placed_objects) >= 3 and sampled_target_bop_objs[0] in placed_objects and visible_forks == 1 and all_scene_visible_forks == 1:
            print(f"Scene {i} qualified with {len(placed_objects)} objects.")
            break
        print(f"Scene {i} did not qualify.")
    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    # Final safety check before camera positioning
    final_visible_forks = [obj for obj in target_forks if not obj.is_hidden()]
    print(f"Final check - visible forks before camera positioning: {[obj.get_name() for obj in final_visible_forks]}")
    if len(final_visible_forks) > 1:
        print("ERROR: Multiple forks are still visible!")
        for obj in final_visible_forks:
            if obj not in sampled_target_bop_objs:
                print(f"Forcibly hiding extra fork: {obj.get_name()}")
                obj.hide(True)



    cam_poses = 0
    while cam_poses < 25:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.35,
                                radius_max = 0.81,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=1, replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'timn1',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    # for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):      
    #     obj.disable_rigidbody()
    #     obj.hide(True)
    # del sampled_target_bop_objs
    # del sampled_distractor_bop_objs
