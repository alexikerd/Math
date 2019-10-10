import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import random
import time
from tqdm import tqdm

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)



replay_memory_size = 50_000
min_replay_memory_size = 1_000
model_name = "50x1"
minibatch_size = 64
discount = 0.99
episodes = 20_000
aggregate_stats_every = 50
epsilon = 0.95
min_epsilon = 0.05
epsilon_decay = 0.9999
keep_prob = 0.2
update_target_every = 8

ohe = preprocessing.OneHotEncoder(categories='auto')
ohe.fit(np.array([['0'],['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9']]).reshape(-1,1))




class ModifiedTensorBoard(TensorBoard):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)


    def set_model(self, model):
        pass


    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)


    def on_batch_end(self, batch, logs=None):
        pass


    def on_train_end(self, _):
        pass


    def update_stats(self, **stats):
        self._write_logs(stats, self.step)









def initiate_state():
    n = ""
    for integer in range(np.random.randint(1,20)):
        n += str(np.random.randint(0,9))
    n = sorted(str(n))
    state = ohe.transform(np.array(sorted(n)).reshape(-1,1)).todense()
    state = np.append(state,np.zeros((20,10)),axis=0)[0:20:1]
    state = np.reshape(state,200,'c')
    return state

def act(state,action):
        
    if(state[0,np.nonzero(action)[0][0]]==1):
        state_prime = state.copy()
        
    else:
        digit,digit_change = int(np.nonzero(action)[1][0]/10), np.nonzero(action)[1][0]%10
        if(digit + 1 > len(np.nonzero(state)[1])):
            if(digit_change==0):
                state_prime = state.copy()
                
            else:
                state_prime = state.copy()
                state_prime[0,(digit*10):(digit*10+10)] = np.zeros((1,10))
                state_prime[0,(digit*10 + digit_change)] = 1
        else:
            state_prime = state.copy()
            state_prime[0,(digit*10):(digit*10+10)] = np.zeros((1,10))
            state_prime[0,(digit*10 + digit_change)] = 1
    number = ''
    for digit in np.nonzero(state_prime)[1][:]:
        number += str(digit%10)
    number = sorted(str(number))
    state_prime = ohe.transform(np.array(sorted(number)).reshape(-1,1)).todense()
    state_prime = np.append(state_prime,np.zeros((20,10)),axis=0)[0:20:1]
    state_prime = np.reshape(state_prime,200,'c')
    return state_prime


def decompose(state):
    number = ''
    for digit in np.nonzero(state)[1][:]:
        number += str(digit%10)
    k = 1
    q = 0
    while(len(str(number))>1):
        q +=1
        k = 1
        for digit in str(number):
            k *= int(digit)
        number = k
    return(q)






class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{model_name}-{int(time.time())}")
        
        self.target_update_counter = 0
    
    def create_model(self):
        model = Sequential()
        model.add(Dense(20,input_dim=200,activation="relu"))
        model.add(Dropout(rate=1-keep_prob))
        
        model.add(Dense(20,input_dim=20,activation="relu"))
        model.add(Dropout(rate=1-keep_prob))
        
        model.add(Dense(200,input_dim=20,activation="linear"))
        model.compile(loss="mse",optimizer=Adam(lr=0.001),metrics=['accuracy'])
        return model
    
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
        
    def get_qs(self,state):
        return self.model.predict(state)[0]
    
    def train(self):
        if len(self.replay_memory) < min_replay_memory_size:
            return
        

        minibatch = random.sample(self.replay_memory,minibatch_size)
        

        
        for index, (state, action, reward, state_prime) in enumerate(minibatch):
            
    
            
            while(np.array_equal(state,state_prime)==False):
                

                
  
            
                future_qs_list = self.target_model.predict(state_prime)
                 
                    
                    
                max_future_q = np.max(future_qs_list[0,:])
                new_q = reward/11 - discount*max_future_q
            
                current_qs = self.model.predict(state)
                current_qs[0,np.nonzero(action)[1][0]] = new_q
                

        
                self.model.fit(state,current_qs,batch_size = minibatch_size, verbose=0,shuffle=False,callbacks=[self.tensorboard])
                
                state = state_prime.copy()
                
                if np.random.uniform(0,1) > epsilon:
                    action = np.zeros((1,200))
                    action[0,np.argmax(agent.get_qs(state))] = 1


                else:    
                    action = np.zeros((1,200))
                    action[0,random.randint(0,199)] = 1
                

                state_prime = act(state,action)
                
                reward = decompose(state_prime) - decompose(state)
                
        
        if self.target_update_counter > update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
  
        else:
            self.target_update_counter += 1
            

            

            

            

            

            
agent = DQNAgent()
epsilon = 0.95

for episode in tqdm(range(1,episodes + 1),ascii=True,unit='episodes'):
    agent.tensorboard.step = episode
    
    reward = 0
    step = 1
    state = initiate_state()
    state_prime = state.copy()
    
    while(np.array_equal(state,state_prime)):
        
        state = initiate_state()        
                
        if np.random.uniform(0,1) > epsilon:
            action = np.zeros((1,200))
            action[0,np.argmax(agent.get_qs(state))] = 1


        else:    
            action = np.zeros((1,200))
            action[0,random.randint(0,199)] = 1

            
        
        state_prime = act(state,action)
        reward = decompose(state_prime) - decompose(state)

    epsilon = max(min_epsilon,epsilon*epsilon_decay)
    

    
    
    agent.update_replay_memory((state, action, reward, state_prime))
    if episode%minibatch_size==0:
        agent.train()








# state = initiate_state()
# action = np.zeros((1,200))
# action[0,np.argmax(agent.get_qs(state))] = 1
# state_prime = act(state,action)

# number = ''
# for digit in np.nonzero(state)[1][:]:
#     number += str(digit%10)


# print(str(number) + ' is ' + str(decompose(state)))


# while(np.array_equal(state,state_prime)==False):
#     state = state_prime.copy()
#     action = np.zeros((1,200))
#     action[0,np.argmax(agent.get_qs(state))] = 1
#     state_prime = act(state,action)
    
    
    
# number2 = ''
# for digit in np.nonzero(state_prime)[1][:]:
#     number2 += str(digit%10)


# print(str(number2) + ' is ' + str(decompose(state_prime)))