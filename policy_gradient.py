import numpy as np
import tensorflow as tf
import gym

n_inputs = 4   #x, xdot, theta, thetadot
n_hidden = 4   #hidden nodes
n_outputs = 1  #probab
learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden_layer = tf.layers.dense(X, n_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden_layer, n_outputs)
outputs = tf.nn.sigmoid(logits)   # P(Left)

probabilties = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial( probabilties, num_samples=1)  #random sampling

y = 1. - tf.to_float(action)   #tensor to float

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients_and_variables = optimizer.compute_gradients(cross_entropy)  #compute gradients to multiply gradients with discount, not minimizing optimizer directly

gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())  #single
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))  #list of tuples

training_operation = optimizer.apply_gradients(grads_and_vars_feed)  #final feed fed

init = tf.global_variables_initializer()
saver = tf.train.Saver()  #to save model later on

def helper_discount_rewards(rewards, discount_rate):  #helper function takes in rewards and applies discount rate, can be 0.95-0.99 etc
    discounted_rewards = np.zeros(len(rewards))       #len(rewards) to 0 in reverse
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):  #takes in all rewards, applies helper_discount function and then normalizes using mean and std deviation
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards,discount_rate))

    flattened_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flattened_rewards.mean()
    reward_std = flattened_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

#Training
env = gym.make("CartPole-v0")   #make the environment

n_game_rounds = 10
max_game_steps = 1000  #time steps before we have to manually day it's done
n_iterations = 500
discount_rate = 0.9

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(n_iterations):
        print("Currently on iteration: {} \n".format(iteration))

        all_rewards = []
        all_gradients = []

        for game in range(n_game_rounds):    #play n game rounds

            current_rewards = []
            current_gradients = []

            obs = env.reset()  #reset environment to default state, that's our first observation

            for step in range(max_game_steps):       #manual cutoff
                cart_pos, cart_vel, pole_ang, ang_vel = obs
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})        #get actions and gradients, reshape comma

                obs, reward, done, info = env.step(action_val[0][0])    #perform action by passing into step function, get observations, reward, boolean indicating whether environment needs to be reset, debug info

                current_rewards.append(reward)          #get current rewards and gradients
                current_gradients.append(gradients_val)

                if done:        #pole fell over (game ended)
                    break

            all_rewards.append(current_rewards)        #append to list of all rewards
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)   #applying helper and normalizing
        feed_dict = {}

        for var_index, gradient_placeholder in enumerate(gradient_placeholders):   #enumerate gives back index locations
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]  #multiplying correct reward with the correct gradient
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        sess.run(training_operation, feed_dict=feed_dict)

    print('Saving graph and session')
    meta_graph_def = tf.train.export_meta_graph(filename='/models/my-policy-gradient-model.meta')  #exporting graphs if we need it in another file
    saver.save(sess, '/models/my-policy-gradient-model')  #saving actual session


#Running this trained model on environment

env = gym.make('CartPole-v0')  #make the environment

obs = env.reset()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('/models/my-policy-gradient-model.meta')
    new_saver.restore(sess,'/models/my-policy-gradient-model')

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])
#end
