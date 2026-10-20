
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    base = '/workspace'
    config = os.path.join(base, 'dual_controllers.yaml')

    env = {
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
            os.environ.get('LD_LIBRARY_PATH', '') + ':' +
            os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', ''),

        'IGN_GAZEBO_RESOURCE_PATH':
            '/home/ubuntu/airobot_ws/install/'
            'crane_plus_description/share:' +
            os.environ.get('IGN_GAZEBO_RESOURCE_PATH', ''),
    }

    gazebo = ExecuteProcess(
        cmd=[
            'ign', 'gazebo', '-r',
            os.path.join(base, 'dual_table.sdf')
        ],
        additional_env=env,
        output='screen'
    )

    actions = [gazebo]

    for robot, x in [('a', '0'), ('b', '0.4')]:
        namespace = 'robot_' + robot
        urdf_path = os.path.join(base, 'crane_' + robot + '.urdf')

        with open(urdf_path, 'r') as f:
            description = f.read()

        actions.append(
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                namespace=namespace,
                parameters=[
                    {'robot_description': description},
                    {'use_sim_time': True}
                ],
                output='screen'
            )
        )

        actions.append(
            TimerAction(
                period=5.0,
                actions=[
                    Node(
                        package='ros_gz_sim',
                        executable='create',
                        arguments=[
                            '-file', urdf_path,
                            '-name', 'crane_' + robot,
                            '-x', x,
                            '-y', '0',
                            '-z', '1.015',
                            '-Y', '3.14159265' if robot == 'b' else '0'
                        ],
                        output='screen'
                    )
                ]
            )
        )

        controllers = [
            'joint_state_broadcaster',
            'crane_plus_arm_controller',
            'crane_plus_gripper_controller'
        ]

        for i, controller in enumerate(controllers):
            args = [
                controller,
                '-c', '/' + namespace + '/controller_manager',
                '-n', '/' + namespace,
                '--controller-manager-timeout', '120'
            ]

            if controller != 'joint_state_broadcaster':
                args += ['-p', config]

            # ロボットごと、コントローラーごとに起動時間をずらす
            delay = 15.0 + (0 if robot == 'a' else 20.0) + i * 5.0

            actions.append(
                TimerAction(
                    period=delay,
                    actions=[
                        Node(
                            package='controller_manager',
                            executable='spawner',
                            arguments=args,
                            output='screen'
                        )
                    ]
                )
            )
    return LaunchDescription(actions)
