import pybullet as p
import numpy as np
from sys import stdin
from time import sleep

# bhand joint info (new):
# (0, b'bhand_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)                  [dummy]
# (1, b'bhand_palm_surface_joint', 4, -1, -1, 0, 0.11, 0.0, 0.0, 3.14, 5.0, 5.0)         [dummy]
# (2, b'bhand_grasp_joint', 4, -1, -1, 0, 0.11, 0.0, 0.0, 3.14, 5.0, 5.0)                [dummy]
# (3, b'finger_1/prox_joint', 0, 7, 6, 1, 0.11, 0.0, 0.0, 3.14, 5.0, 5.0)                [action 0]
# (4, b'finger_1/med_joint', 0, 8, 7, 1, 0.11, 0.0, 0.0, 2.44, 5.0, 5.0)                 [action 1]
# (5, b'finger_1/dist_joint', 0, 9, 8, 1, 0.11, 0.0, 0.0, 0.83, 5.0, 5.0)                [action 1]
# (6, b'finger_1/tip_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)                [dummy]
# (7, b'finger_2/prox_joint', 0, 10, 9, 1, 0.11, 0.0, 0.0, 3.14, 5.0, 5.0)               [action 0]
# (8, b'finger_2/med_joint', 0, 11, 10, 1, 0.11, 0.0, 0.0, 2.44, 5.0, 5.0)               [action 2]
# (9, b'finger_2/dist_joint', 0, 12, 11, 1, 0.11, 0.0, 0.0, 0.83, 5.0, 5.0)              [action 2]
# (10, b'finger_2/tip_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)               [dummy]
# (11, b'finger_3/med_joint', 0, 13, 12, 1, 0.11, 0.0, 0.0, 2.44, 5.0, 5.0)              [action 3]
# (12, b'finger_3/dist_joint', 0, 14, 13, 1, 0.11, 0.0, 0.0, 0.83, 5.0, 5.0)             [action 3]
# (13, b'finger_3/tip_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)               [dummy]

ACTIONS = [[3, 7], [4, 5], [8, 9], [11, 12]]

# action meaning:
# action 0    - rotate 2 moving fingers simultaneously
# action 1    - bend 1st finger
# action 2    - bend 2nd finger
# action 3    - bend 3rd finger
# action 4:7  - set linear speed of the hand
# action 7:10 - set angular speed of the hand

class GraspEnvExample:
    def __init__(self):
        self._physId = p.connect(p.GUI)
        self._k = 4
        self._init_params = {
            'hand_urdf': 'barrett_model/bhand.urdf',
            'hand_pos': [0, 0, 1],
            'hand_orient': [0, 3.14, 0],
            'object_urdf': 'cube_my.urdf',
            'object_pos': [0, 0, 0.07],
            'object_orient': [0, 0, 0],
            'floor_urdf': 'plane100.urdf',
        }
        self._bhand_id = self._init_hand()
        self._obj_id = self._init_scene()
        self._torques_force = 5


    def _init_hand(self):
        ''' Add the Barrett hand to the scene.

        Return:
            bhand_id (int): Id returned by pybullet loadURDF. '''
        return p.loadURDF(self._init_params['hand_urdf'],
                          self._init_params['hand_pos'],
                          p.getQuaternionFromEuler(self._init_params['hand_orient']),
                          physicsClientId=self._physId)


    def _init_scene(self):
        ''' Add objects and the plane to the scene. All the init parameters
        should be placed in _init_params'''
        plane_id = p.loadURDF(self._init_params['floor_urdf'], physicsClientId=self._physId)
        return p.loadURDF(self._init_params['object_urdf'],
                          self._init_params['object_pos'],
                          p.getQuaternionFromEuler(self._init_params['object_orient']),
                          physicsClientId=self._physId)


    def step(self, action):
        ''' Do _k steps with the specified action, return observation and reward after the last one.
        Args:
        action (1D float):            Value from actions_space.
        '''
        for _ in range(self._k):
            # first apply the gravity force to all objects
            p.applyExternalForce(self._obj_id,
                                -1,
                                [0,0,-9.8],
                                [0,0,0],
                                p.LINK_FRAME,
                                physicsClientId=self._physId)

            # check whether action length corresponds to action_space
            if len(action) != 10:
                raise ValueError("action should have length {}".format(10))

            # group some joints in order to resemble real bhand behavior
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[0][0],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[0],
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[0][1],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[0],
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[1][0],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[1],
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[1][1],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[2][0],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[2],
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[2][1],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[3][0],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[3],
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.setJointMotorControl2(self._bhand_id,
                                    ACTIONS[3][1],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=self._torques_force,
                                    physicsClientId=self._physId)
            p.resetBaseVelocity(self._bhand_id,
                                linearVelocity=np.array(action[-6:-3]),
                                angularVelocity=np.array(action[-3:]),
                                physicsClientId=self._physId)

            # after all the actions are set, impose hard constraints
            p.stepSimulation(self._physId)


    def run(self, ask_user=False, nb_iter=300):
        action = [0, 0.20, 0.20, 0.20, 0, 0, -0.145, 0, 0, 0]
        for i in range(nb_iter):
            if i % 100 == 0 and ask_user:
                print('[{}] Please type the next action (7 numbers separated by space)'.format(i))
                action_string = []
                while len(action_string) != 7:
                    action_string = stdin.readline().split()
                action_short = [float(x) for x in action_string]
                action = action_short + [0,0,0]
                print('action set = {}'.format(action))

            self.step(action[:])
        sleep(10)


if __name__ == "__main__":
    env = GraspEnvExample()
    env.run()
