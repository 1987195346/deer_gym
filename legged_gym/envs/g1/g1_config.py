from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 168    # 47 + 121 = 168, 121是地形的观测点
        num_privileged_obs = 50
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True  #地形课程学习如果打开，机器人重置后会放在最左边的地面
        # rough terrain only:
        measure_heights = True
        #11 x 11 = 121 points
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 0.9         # #DOF软限制：设置一个较小的活动范围，以防止 DOF 迅速达到极限，从而提高训练的稳定性
        base_height_target = 0.78        # 目标底座高度
        feet_height_target = 0.08        # 目标抬脚高度

        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.0          #机器人在Z轴方向上的线性速度进行惩罚（非水平要改），uni是-2，leg是-0.0
            ang_vel_xy = -0.05
            orientation = -0.1        #惩罚机器人基础姿态（即机器人身体的朝向）偏离平坦的情况（非水平需要改），uni是-1，leg是-0.1
            torques = -0.00001
            dof_vel = -1e-3           #惩罚关节速度，防止运动过快，uni是-1e-3，leg是0
            dof_acc = -2.5e-7
            base_height = -0.0       #惩罚底座高度和目标底座高度差异（非水平可以改），uni是-10，leg是-0
            feet_air_time = 0.0       #对机器人脚部在空中的时间进行奖励,鼓励迈大步，uni是0，leg是5.0
            collision = -1.0           #惩罚机器人与环境中的物体发生碰撞，uni是0，leg是-1
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -5.0     #对接近关节位置极限的情况进行惩罚uni是-5，leg是-1
            # 下面的奖励项是g1中的
            alive = 0.15              #g1new  奖励机器人处于活跃状态
            hip_pos = -1.0            #g1new  惩罚机器人的髋关节当前位置与默认位置的偏差(不清楚是否需要改)，uni是-1
            contact_no_vel = -0.2     #g1new  惩罚脚无速度接触地面，使走路更平滑
            feet_swing_height = -0.0  #g1new  惩罚每个脚部在 z 方向上的位置与理想摆动高度的偏差（非水平要改，因为这个高度很难跨越台阶，uni是-20.0）
            contact = 0.18            #g1new  奖励脚部的接触状态与支撑态状态一致情况
            no_fly = 0.25            #cassie 奖励机器人正好有一个脚在地面上

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 5000
        run_name = ''
        experiment_name = 'g1'
        save_interval = 1000 # check for potential saves every this many iterations