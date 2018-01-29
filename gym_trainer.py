import gym
import tensorflow as tf
import random
import numpy as np

from statistics import median, mean
from collections import Counter



LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000




def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(200):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print ("action:%s - obser\Cposition:%s obser\Cvelocity:%s obser\Pangle:%s obser\PTvelocity:%s - reward:%s - done:%s - info:%s" % (action,observation[0],observation[1],observation[2],observation[3],reward,done,info))
            if done:
                break

def init_game():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
#                training_data.append([data[0], output])
                training_data.append([data[0], data[1]])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)

# some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data


def model(inp,outp):
    classes =2

    feature_columns = [ tf.feature_column.numeric_column('x',shape=[4])]

    regressor = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[4,1],n_classes=2,model_dir="/tmp/gym")

    train_input_fun = tf.estimator.inputs.numpy_input_fn(x={'x':inp},y=outp,batch_size=1000,num_epochs=None,shuffle=True)
    regressor.train(input_fn=train_input_fun,steps=2000)

    return regressor



data = init_game()

c = np.array([i[0] for i in data])
d = np.array([i[1] for i in data])

tf.reset_default_graph()


m = model(c,d)
print m




scores = []
choices = []
for each_game in range(100):
    print "game on:# "+str(each_game)
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            obser = np.array(prev_obs)
            obser = np.vstack((obser, prev_obs[prev_obs[:,] < 10000]))
            p_inp_fun= tf.estimator.inputs.numpy_input_fn(x={"x":obser},num_epochs=1,shuffle=False)
            p = m.predict(input_fn=p_inp_fun)


            for i in p:
                action = int(i['classes'][0])
            #print action
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done:
            break
            print "game ended"
    print score
    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
