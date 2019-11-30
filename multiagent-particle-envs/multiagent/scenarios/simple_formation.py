import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 0
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.08
        # add edges
        world.edges = []
        for i in range(len(world.agents)):
            world.edges.append(1)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i in range(len(world.edges)):
            world.edges[i] = np.random.uniform(0.4, 1.7, 1)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # for i in range(len(world.edges)):
        #     print(world.edges[i])


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.05:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 3
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min + 0.01 else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)

        # index = world.agents.index(agent)
        # if index = len(world.agents) - 1:
        #     rew -= abs(np.sqrt(np.sum(np.square(world.agents[index-1].state.p_pos - agent.state.p_pos))) - world.edges[index-1]) / 2
        #     rew -= abs(np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - agent.state.p_pos))) - world.edges[index]) / 2
        # else:
        #     rew -= abs(np.sqrt(np.sum(np.square(world.agents[index-1].state.p_pos - agent.state.p_pos))) - world.edges[index-1]) / 2
        #     rew -= abs(np.sqrt(np.sum(np.square(world.agents[index+1].state.p_pos - agent.state.p_pos))) - world.edges[index]) / 2            
        dists = []
        for i in range(len(world.agents)):
            dists.append(abs(np.sqrt(np.sum(np.square(world.agents[i-1].state.p_pos - world.agents[i].state.p_pos))) - world.edges[i-1]))
            rew -= min(dists)
            # rew -= 10*abs(np.sqrt(np.sum(np.square(world.agents[i-1].state.p_pos - world.agents[i].state.p_pos))) - world.edges[i-1])
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 3
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for agent in world.agents:
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= 2*bound(x)

        return rew

    def plot_data(self, world):
        plot_data = []
        plot_data.append([np.sqrt(np.sum(np.square(world.agents[i-1].state.p_pos - world.agents[i].state.p_pos))) for i in range(len(world.agents))])
        plot_data.append([world.edges[i-1] for i in range(len(world.agents))])
        return plot_data

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
