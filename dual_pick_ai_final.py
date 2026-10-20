
#!/usr/bin/env python3
import math
import pickle
import random
import threading
import time
from collections import deque

import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [f'crane_plus_joint{i}' for i in range(1, 5)]


class DualPickAI(Node):
    def __init__(self):
        super().__init__('dual_pick_ai')

        with open('/workspace/intent_model.pkl', 'rb') as f:
            saved = pickle.load(f)
        self.model = saved['model']
        self.features = saved['features']

        self.workspace_cond = threading.Condition()
        self.workspace_owner = None
        self.waiting = set()
        self.request_time = {}


        self.state_lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {'a': 'IDLE', 'b': 'IDLE'}
        self.scores = {'a': None, 'b': None}
        self.history = {'a': deque(), 'b': deque()}

        self.arm = {}
        self.hand = {}
        self.intent_clients = {}

        for robot in ('a', 'b'):
            ns = f'/robot_{robot}'
            self.arm[robot] = self.create_publisher(
                JointTrajectory,
                f'{ns}/crane_plus_arm_controller/joint_trajectory', 10
            )
            self.hand[robot] = self.create_publisher(
                JointTrajectory,
                f'{ns}/crane_plus_gripper_controller/joint_trajectory', 10
            )
            self.create_subscription(
                JointState, f'{ns}/joint_states',
                lambda msg, r=robot: self.observe(r, msg), 10
            )
            self.create_service(
                Trigger, f'{ns}/ask_intent',
                lambda req, res, r=robot: self.answer(r, res)
            )
            self.intent_clients[robot] = self.create_client(
                Trigger, f'{ns}/ask_intent'
            )

    def answer(self, robot, response):
        with self.state_lock:
            state = self.state[robot]
        response.success = state in ('REQUEST', 'PICKING')
        response.message = state
        return response

    def observe(self, robot, msg):
        positions = dict(zip(msg.name, msg.position))
        if not all(j in positions for j in JOINTS):
            return

        now = time.monotonic()
        angles = tuple(math.degrees(positions[j]) for j in JOINTS)
        history = self.history[robot]
        history.append((now, angles))

        while len(history) > 2 and history[1][0] <= now - 0.3:
            history.popleft()

        if now - history[0][0] < 0.3:
            return

        target = now - 0.3
        past = history[0][1]
        for t, a in history:
            if t <= target:
                past = a
            else:
                break

        deltas = [x - y for x, y in zip(angles, past)]
        x = pd.DataFrame(
            [list(angles) + deltas], columns=self.features
        )
        probs = self.model.predict_proba(x)[0]
        score = float(probs[list(self.model.classes_).index(1)])

        with self.state_lock:
            self.scores[robot] = score

    def command(self, pub, joints, positions, seconds):
        if self.stop.is_set() or not rclpy.ok():
            return
        msg = JointTrajectory()
        msg.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int(
            (seconds % 1) * 1e9
        )
        msg.points = [point]
        if self.stop.is_set() or not rclpy.ok():
            return

        try:
            pub.publish(msg)
        except rclpy.executors.ExternalShutdownException:
            return
        except Exception as e:
            if self.stop.is_set() or not rclpy.ok():
                return
            raise
        self.stop.wait(seconds + 0.3)

    def move(self, robot, degrees):
        self.command(
            self.arm[robot], JOINTS,
            [math.radians(v) for v in degrees], 2.0
        )

    def gripper(self, robot, angle):
        self.command(
            self.hand[robot],
            ['crane_plus_joint_hand'], [angle], 0.8
        )

    def pick(self, robot):
        self.move(robot, [0, 0, 0, 0])
        target = [-15, 25, -35, 10] if robot == 'a' else \
                 [15, 25, -35, 10]
        self.move(robot, target)
        self.gripper(robot, 0.25)
        self.gripper(robot, -0.15)
        self.move(robot, [0, 0, 0, 0])

    def acquire_workspace(self, robot):
        logged_yield = False
        with self.workspace_cond:
            self.waiting.add(robot)
            self.request_time[robot] = time.monotonic()

            while not self.stop.is_set():
                # 同時に待っている場合、要求時刻の早い方を優先。
                # 同時刻ならAを優先。
                winner = min(
                    self.waiting,
                    key=lambda r: (
                        self.request_time[r],
                        0 if r == 'a' else 1
                    )
                )

                if self.workspace_owner is None and winner == robot:
                    self.workspace_owner = robot
                    self.waiting.remove(robot)
                    self.request_time.pop(robot, None)
                    return True

                if not logged_yield:
                    self.get_logger().info(
                        f'{robot.upper()} YIELD: workspace reserved'
                    )
                    logged_yield = True

                self.workspace_cond.wait(timeout=0.2)

            self.waiting.discard(robot)
            self.request_time.pop(robot, None)
            self.workspace_cond.notify_all()
            return False

    def release_workspace(self, robot):
        with self.workspace_cond:
            if self.workspace_owner == robot:
                self.workspace_owner = None
                self.workspace_cond.notify_all()

    def worker(self, robot):
        other = 'b' if robot == 'a' else 'a'

        for cycle in range(3):
            if self.stop.is_set():
                return

            with self.state_lock:
                self.state[robot] = 'IDLE'

            delay = random.uniform(1.0, 5.0)
            self.get_logger().info(
                f'{robot.upper()} IDLE {delay:.1f}s'
            )
            if self.stop.wait(delay):
                return

            with self.state_lock:
                self.state[robot] = 'REQUEST'
                score = self.scores[other]

            self.get_logger().info(
                f'{robot.upper()} REQUEST | '
                f'AI predicts {other.upper()} PICK: {score}'
            )

            # AI推定が高ければ、相手へServiceで確認
            if score is not None and score >= 0.40:
                future = self.intent_clients[other].call_async(
                    Trigger.Request()
                )
                deadline = time.monotonic() + 2.0
                while not future.done() and time.monotonic() < deadline:
                    if self.stop.wait(0.05):
                        return

                if future.done():
                    try:
                        result = future.result()
                        self.get_logger().info(
                            f'{robot.upper()} SERVICE CONFIRMED: '
                            f'{other.upper()} state={result.message}'
                        )



                        if result.success:
                            self.get_logger().info(
                                f'{robot.upper()} SERVICE: '
                                f'{other.upper()} intends to PICK '
                                f'({result.message})'
                            )

                            # Serviceは相手の意図確認に使用する。
                            # PICKの実行順序は中央のworkspace調停機能が決定する。
                            self.get_logger().info(
                                f'{robot.upper()} WAIT FOR WORKSPACE'
                            )

                            

                    except Exception as e:
                        self.get_logger().warn(str(e))

            # AIの誤判定があっても同時進入は禁止
            if not self.acquire_workspace(robot):
                return

            try:
                if self.stop.is_set():
                    return
                with self.state_lock:
                    self.state[robot] = 'PICKING'
                self.get_logger().info(
                    f'{robot.upper()} PICK START'
                )
                self.pick(robot)
                if not self.stop.is_set() and rclpy.ok():
                    self.get_logger().info(
                        f'{robot.upper()} PICK END'
                    )
            finally:
                with self.state_lock:
                    self.state[robot] = 'IDLE'
                self.release_workspace(robot)


def main():
    rclpy.init()
    node = DualPickAI()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    ros_thread = threading.Thread(
        target=executor.spin, daemon=True
    )
    ros_thread.start()

    workers = [
        threading.Thread(target=node.worker, args=(r,))
        for r in ('a', 'b')
    ]

    try:
        time.sleep(2)
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

    except KeyboardInterrupt:
        node.get_logger().info('Stopping demo...')
        node.stop.set()
    finally:
        node.stop.set()

        # ワーカーの終了を待ってからROSを終了する
        for worker in workers:
            if worker.ident is not None:
                worker.join()

        executor.shutdown()
        ros_thread.join()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()
