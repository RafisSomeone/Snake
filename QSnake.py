import math
import random
import gym
from operator import itemgetter
import numpy as np
from pygame.constants import NOEVENT
from ple.games import snake
from ple.games.snake import Snake
import matplotlib.pyplot as plt
import threading
import functools
import pickle
import sys
import csv
import pygame

from ple.ple import PLE
snake_length = int(sys.argv[1]) 

class QLearner:
    def __init__(self):
        filename = f'New_Length{snake_length}compact_QSnakeData.pickle'
        try:
            with open(filename, "rb") as file:
                self.Q = pickle.load(file)
        except:
            self.Q = {}
        fps = 30  # fps we want to run at
        frame_skip = 2
        num_steps = 1
        force_fps = False  # slower speed
        display_screen = True
        self.length_snaking = 1
        reward = 0.0
        max_noops = 20
        nb_frames = 15000
        self.game = Snake(width=400,height=400, init_length=snake_length)  
        self.p = PLE(self.game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)
        self.attempt_no = 1
        self.actions = self.p.getActionSet()
        self.width = self.p.getScreenDims()[0]
        self.height = self.p.getScreenDims()[1]
        print(f'actions {self.actions}')
        print(f'state {self.p.getGameState()}')
        print(f'dims {self.p.getGameStateDims()}')
        print(f'dims {self.p.getScreenDims()}')
        print(f'screen {self.actions[np.random.randint(0, len(self.actions))]}')

    def learn(self, max_attempts,results,j, sigma, alpha, epsilon):
        result = []
        infinite = True
        for i in range(max_attempts):
            reward_sum = self.attempt(i, max_attempts,sigma, alpha, epsilon)
            with open(f'Reward{snake_length}compacted.csv', 'a+', encoding='UTF8') as f:
                f.write(str(reward_sum) + '\n')
            result.append(reward_sum)
            if infinite:
                i -= 1
        results[j] = result

    def get_reward(self, observation,new_observation,reward):
        head_food_distance = 3
        old_distnace = observation[0]
        new_distance = new_observation[0]
        if new_distance < old_distnace:
            reward += head_food_distance
        else: 
            reward -= head_food_distance
        return reward

    def attempt(self, attempt_i, max_attempts, sigma_s, alpha_s, epsilon_s):
        half_attempt = 0
        epsilon = epsilon_s
        sigma = sigma_s
        alpha = alpha_s
        filename = f'New_Length{self.game.player.length}compact_QSnakeData.pickle'
        if attempt_i % 1000 == 0:
            with open(filename,'wb' ) as file:
                pickle.dump(dict(self.Q), file)
            print("+++++++++ Pickle saved +++++++++")
        # print(half_attempt > attempt_i)
        if half_attempt > attempt_i:
            epsilon = 1.0 - attempt_i / half_attempt
            alpha = 1.0 - attempt_i / half_attempt
        else:
            self.game.isLostOn = False
        self.game.isLostOn = False
        oldState = self.p.getGameState()
        observation = self.discretise(oldState,None)
        done = False
        reward_sum = 0.0
        index = 0 
        while not done:
            filename = f'New_Length{self.game.player.length}compact_QSnakeData.pickle'
            try:
                with open(filename, "rb") as file:
                    print(filename)
                    self.Q = pickle.load(file)
            except:
                print('error')
                self.Q = {}
            action = self.pick_action(observation,epsilon)
            reward = self.p.act(action)
            state = self.p.getGameState()
            new_observation = self.discretise(state, oldState)
            oldState = state
            reward = self.get_reward(observation,new_observation,reward)
            self.update_knowledge(action, observation, new_observation, reward, alpha, sigma)
            # self.update_knowledge_SARSA(action, observation, new_observation, reward, alpha, sigma,epsilon)
            observation = new_observation
            reward_sum += reward
            if self.p.game_over():
                self.p.reset_game()
                done = True
            index += 1 
            pygame.image.save(self.game.screen, f"screen_all_{index}.jpeg")
        self.attempt_no += 1
        return reward_sum

    def get_bucket(self, value, buckets, lower_bound, interval):
        for i in range(1, buckets):
            tmp = lower_bound + (i * interval)
            if value < tmp:
                return i
            i += 1
        return buckets

    def discretise_value(self, value,upper_bounds,lower_bounds):
        buckets = 20
        interval = (upper_bounds - lower_bounds) / float(buckets)
        return self.get_bucket(value, buckets, lower_bounds, interval)

    def get_food_head_distance_and_vector_from_state(self, state):
        return self.get_x_y_head_distance_and_vector_from_state(state,state['food_x'],state['food_y'])
        
        
    def get_x_y_head_distance_and_vector_from_state(self, state,x ,y):
        snake_head_point = np.array([state['snake_head_x'],state['snake_head_y']])
        head_food_vector = snake_head_point - np.array([x ,y])
        return np.linalg.norm(head_food_vector), head_food_vector

    def get_direction_vector_from_x_y(self,x,y):
        direction = ((round(math.atan2(y,x) / (2 * math.pi / 8.0))) + 8) % 8
        direction_map = {
            0: (0,0,0,1), #up, down, left, right
            1: (1,0,0,1),
            2: (1,0,0,0),
            3: (1,0,1,0),
            4: (0,0,1,0),
            5: (0,1,1,0),
            6: (0,1,0,0),
            7: (0,1,0,1)
        }
        return direction_map[direction]

    def get_direction_vector_for_obstacle_from_x_y(self,x,y):
        direction = ((round(math.atan2(y,x) / (2 * math.pi / 8.0))) + 8) % 8
        direction_map = {
            0: (0,0,0,1), #up, down, left, right
            1: (0,0,0,0),
            2: (1,0,0,0),
            3: (0,0,0,0),
            4: (0,0,1,0),
            5: (0,0,0,0),
            6: (0,1,0,0),
            7: (0,0,0,0)
        }
        return direction_map[direction]
    def discretise(self, state, oldState):
        
        # angle = self.discretise_value(observation[2],2)
        # angular_velocity = self.discretise_value(observation[3],3)
        # return 0, cart_velocity, angle, angular_velocity
        head_food_distance, head_food_vector = self.get_food_head_distance_and_vector_from_state(state)

        isNearer = 0
        if oldState is None: 
            isNearer = 2
        else:
            old_Distance, _old_vector = self.get_food_head_distance_and_vector_from_state(oldState)
            if head_food_distance < old_Distance:
                isNearer = 1

        norm_vector = head_food_vector/head_food_distance

        distance_from_obstackle = 30
        distance_from_wall = 30
        x_head = state['snake_head_x']
        y_head  = state['snake_head_y']
        head_vectors = []

        if(x_head - distance_from_wall <= 0):
            head_vectors.append((0,0,1,0))
        
        if(x_head + distance_from_wall >= self.width):
            head_vectors.append((0,0,0,1))

        if(y_head - distance_from_wall <= 0):
            head_vectors.append((1,0,0,0))
        if(y_head + distance_from_wall >= self.height):
            head_vectors.append((0,1,0,0))

        snake_body_positions = np.array(state['snake_body_pos'])

        snake_body = [] 
        skip = 0
        for body in snake_body_positions:
            if (skip>2):
                snake_body.append(body)
            skip+=1

        snake_body_distance_and_vectors = map(lambda x: self.get_x_y_head_distance_and_vector_from_state(state,x[0],x[1]),snake_body)
        snake_body_positions_less_than_obstacle = filter(lambda x: x[0] < distance_from_obstackle, snake_body_distance_and_vectors)
        snake_body_vectors = list(map(lambda x: self.get_direction_vector_for_obstacle_from_x_y(x[1][0],x[1][1]),snake_body_positions_less_than_obstacle))
        snake_body_vectors += head_vectors
        obstacle_tuple = (0,0,0,0)
        if(len(snake_body_vectors) != 0):
            obstacle_tuple = functools.reduce(lambda a, b: np.array(a)+np.array(b), snake_body_vectors)
            obstacle_tuple = tuple(list(map(lambda x: 1 if x > 1 else x,obstacle_tuple)))

        # print(self.game.player.body[0].pos.x)
        # upper_boundry_distance = math.dist((0,0),self.p.getScreenDims())
        # lower_boundry_distance = 0
        # distance_head_food = self.discretise_value(head_food_distance, upper_boundry_distance, lower_boundry_distance)
        
        return (isNearer,) + self.get_direction_vector_from_x_y(norm_vector[0], norm_vector[1]) + (self.game.player.dir.x,self.game.player.dir.y) + obstacle_tuple
    def pick_action(self, observation,eps):
        if random.random() < eps:
            return self.actions[np.random.randint(0, len(self.actions))]
        else:  
            tmp = [(self.actions[0],-500)]
            for key, value in self.Q.items():
                if key[0] == observation:
                    tmp.append((key[1],value))
            return max(tmp,key=itemgetter(1))[0]
        
    def update_knowledge(self, action, observation, new_observation, reward, alpha, sigma):
        tmp = [(self.actions[0],-500)]
        for key, value in self.Q.items():
            if key[0] == new_observation:
                tmp.append((key[1],value))
        Qnew = max(tmp,key=itemgetter(1))[1]
        self.Q[(observation,action)] = (1 - alpha) * self.Q.get((observation,action),0) + alpha * (reward + sigma * Qnew)
    
    def update_knowledge_SARSA(self, action, observation, new_observation, reward, alpha, sigma, epsilon):
        self.Q[(observation,action)] = self.Q.get((observation,action),0) + alpha * (reward + sigma * self.Q.get((new_observation,self.pick_action(new_observation,epsilon)),0) - self.Q.get((observation,action),0))
        
def moving_average(x, w=10):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    sigmas = [0.99] 
    alphas = [0.01]
    epsilons = [0.01]
    results = []
    QLearner().learn(3000000,results,0,sigmas[0],alphas[0],epsilons[0])
    # for s in range(len(sigmas)):
    #     max_attempts = 3000
    #     thread_number = 1
    #     threads = [[]] * thread_number
    #     lerners = [[]] * thread_number
    #     results = [[]] * thread_number
    #     for i in range(len(threads)):
    #         lerners[i] = QLearner()
    #         threads[i] = threading.Thread(target=lerners[i].learn, args=(max_attempts,results,i,sigmas[s],alphas[s],epsilons[s]))
    #         threads[i].start()

        # for i in range(len(threads)):
        #     threads[i].join()
        #     with open(f'sarsa/Results_{s}_{i}.csv', 'w+', encoding='UTF8') as f:
        #         for reward in results[i]:
        #             f.write(str(reward) + '\n')
        # moving_average_result = moving_average(results[1])

        # plt.plot(moving_average_result)
        # plt.ylabel('Reward')
        # plt.xlabel('Attempt')
        # plt.title(f'Moving average sigma = {sigmas[s]}, alpha = {alphas[s]}, epsilon = {epsilons[s]}')
        # plt.savefig(f'sarsa/TheNewest_buckets_moving_average_{s}.png')
        # plt.show(block = False)
        # plt.clf()

        # avg = np.average(results, axis=0)
        # std = np.std(results, axis=0)
        # plt.plot(range(len(avg)), avg)
        # plt.fill_between(range(len(avg)), avg-std, avg+std,alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        # linewidth=4, linestyle='dashdot', antialiased=True)
        # plt.ylabel('Reward')
        # plt.xlabel('Attempt')
        # plt.title(f'Average sigma = {sigmas[s]}, alpha = {alphas[s]}, epsilon = {epsilons[s]}')
        # plt.savefig(f'sarsa/TheNewest_buckets_average_{s}.png')
        # plt.show(block = False)
        # plt.clf()


if __name__ == '__main__':
    main()