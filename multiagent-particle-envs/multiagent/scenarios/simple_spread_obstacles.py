import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

size_agents = 0.15
size_landmarks = 0.05
size_obstacles = 0.20

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        num_obstacles = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = size_agents
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = size_landmarks
        # add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.obstacles):
            landmark.name = 'obstacles %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = size_obstacles
            landmark.boundary = False
        world.landmarks += world.obstacles
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.65])
        # random properties for obstacles
        for i, landmark in enumerate(world.obstacles):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # make sure that the generated obstacles are not too close to any target
        for i, obstacle in enumerate(world.obstacles):
            if not obstacle.boundary:
                while(True):          
                    obstacle.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
                    obstacle.state.p_vel = np.zeros(world.dim_p)
                    dists = []
                    for landmark in world.landmarks:
                        if landmark not in world.obstacles:
                            dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - obstacle.state.p_pos))))
                    if min(dists) >= size_obstacles + size_agents:
                        break


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            if l not in world.obstacles:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.1:
                    occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision_agent(a, agent):
                    rew -= 1
                    collisions += 1
            for b in world.obstacles:
                if b.collide:
                    if self.is_collision_obstacle(b, agent):
                        rew -= 1
                        collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision_agent(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_collision_obstacle(self, agent, obstacle):
        delta_pos = agent.state.p_pos - obstacle.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + obstacle.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            if l not in world.obstacles:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision_agent(a, agent):
                    rew -= 1
            for b in world.obstacles:
                if b.collide:
                    if self.is_collision_obstacle(b, agent):
                        rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
