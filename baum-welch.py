import re
from copy import deepcopy
from random import random

def initialize_pi(tags):
    initial_prob = dict()
    total = 0.0
    for tag in tags:
        initial_prob[tag] = random()
        total += initial_prob[tag]

    initial_prob = dict((tag,(initial_prob[tag])/total) for tag in tags)
    return initial_prob

def calc_gamma(alpha, beta, tags, observation):
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1

    gamma = dict()
    for i in range(1, len(observation)+1):
        gamma[i] = dict()

    for i in range(1, len(observation)+1):
        for tag in tags:
            gamma[i][tag] = (beta[i][tag] * alpha[i][tag]) / alpha[len(observation)+1]
            print("ALPHA", alpha[i][tag] , beta[i][tag] , alpha[len(observation)+1])

    return gamma

def calc_eta(a_matrix, b_matrix, alpha, beta, tags, observation):
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1
    eta = dict()
    for i in range(1, len(observation)+1):
        eta[i] = dict()
        for tag in tags:
            eta[i][tag] = dict()
    
    for i in range(1, len(observation)):
        for tag1 in tags:
            for tag2 in tags:
                eta[i][tag1][tag2] = (alpha[i][tag1] * a_matrix[tag1][tag2] * beta[i+1][tag2] * b_matrix[tag2][observation[i]])/alpha[len(observation)+1]
    return eta

def initialize_a(tags):
    a = dict()
    for i in tags:
        for j in tags:
            if i not in a:
                a[i] = dict()
            a[i][j] = random()
        a[i]['f'] = random()
    return a

def initialize_b(tags, line_list):
    b = dict()
    for sentence in line_list:
        for word in sentence:
            if word == '':
                continue
            else:
                for i in tags:
                    if i not in b:
                        b[i] = dict()
                    b[i][word] = random()
    return b

def normalize_a(a, tags):
    a_matrix = deepcopy(a)
    total = 0.0
    for tag1 in tags:
        for tag2 in tags:
            total = total + a_matrix[tag1][tag2] 
        total = total + a_matrix[tag1]['f']

    for tag1 in tags:
        for tag2 in tags:
            a_matrix[tag1][tag2] = (a_matrix[tag1][tag2])/total
        a_matrix[tag1]['f'] = (a_matrix[tag1]['f'])/total
        # print("normalisied A:", tag1, a_matrix[tag1]['f'])

    return a_matrix

def inlayer_norm_b(b, tag_list, observation):
    for word in observation:
        if word == '':
            continue
        else:
            total = 0.0
            for i in tag_list:
                total += b[i][word]
            for i in tag_list:
                b[i][word] = (b[i][word])/total
    return b

def normalize_b(b, tags, sentences):
    
    for sentence in sentences:
        for word in sentence:
            if word == '':
                continue
            else:
                total = 0.0
                for tag in tags:
                    total += b[tag][word]
                for tag in tags:
                    b[tag][word] = (b[tag][word])/total
    # TC
    for sentence in sentences:
        for word in sentence:
            if word == '':
                continue
            else:
                1
                #print("b", i, word, b[i][word])
    return b

def backward(a_matrix, b_matrix, pi, observation, tags):       # beta[timestamp][tag]

    beta = dict()
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1
    # print(observation)

    for i in range(1,len(observation)+1):
        beta[i] = dict()
    # Initialize the T timestamp probs
    for tag in tags:
        beta[len(observation)][tag] = a_matrix[tag]['f']

    for i in range(len(observation)-1, 0, -1):
        j = i+1
        for tag_pres in tags:
            beta[i][tag_pres] = 0.0
            for tag_future in tags:  
                # print(type(tag_future),tag_future,type(observation[i]),observation[i])
                # print(b_matrix[tag_future][observation[i]]) 
                beta[i][tag_pres] += (beta[j][tag_future] * a_matrix[tag_pres][tag_future] * b_matrix[tag_future][observation[i]])
    #Final layer computation
    final = 0
    beta[final] = 0.0
    for tag in tags:
        beta[final] += (beta[final+1][tag] * pi[tag])
    return beta

# def pos_tags():
#     tags = ['NP', 'NN', 'JJ', 'IN', 'VB', 'TO', 'DT', 'PRP', 'RB', 'CC']
#     return tags

def forward(a_matrix, b_matrix, pi, observation, tags):        # alpha[timestamp][tag]
    alpha = dict()
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1

    for i in range(1,len(observation)+1):
        alpha[i] = dict()
    # initialize
    alpha[1] = dict((tag,(pi[tag]*b_matrix[tag][observation[0]])) for tag in tags)
    for i in range(2, len(observation)+1):
        j = i-1
        for tag_pres in tags:
            alpha[i][tag_pres] = 0.0
            for tag_prev in tags: 
                alpha[i][tag_pres] += (alpha[j][tag_prev] *  b_matrix[tag_pres][observation[j]] * a_matrix[tag_prev][tag_pres])

    #Final layer computation
    final = len(observation)+1
    alpha[final] = 0.0
    for tag in tags:
        alpha[final] += (a_matrix[tag]['f']*alpha[final-1][tag])
        # print("alpha[x]",x,tag,alpha[final-1][tag],a_matrix[tag]['f'],alpha[final])
    return alpha

def baum_welch(a_matrix, b_matrix, tags, line_list):
    # observation = line_list[0]
    for k,observation in enumerate(line_list):
        if k == 20:
            break
        elif observation == '':
            continue
        else:
            print("Iteration number:",k)
            print("Sentence number:",observation)
        
        for i, word in enumerate(observation):
            if word == '':
                del observation[i]
            else:
                1
        
        for i in range(10):     # Fixed number of observations = 1000
            print("-----------",i,"------------")
            #E-STEP
            beta = backward(a_matrix, b_matrix, pi, observation, tags)
            alpha = forward(a_matrix, b_matrix, pi, observation, tags)
            
            eta = calc_eta(a_matrix, b_matrix, alpha, beta, tags, observation)
            gamma = calc_gamma(alpha, beta, tags, observation)
            
            #M-STEP
            prev_b_matrix = deepcopy(b_matrix) 
            prev_a_matrix = deepcopy(a_matrix)

            for tag in tags:
                for word in observation:
                    numer = denom = 0.0
                    for t in range(1, len(observation)):
                        if observation[t] == word:
                            numer += gamma[t][tag]
                        else:
                            numer += 0
                        denom += gamma[t][tag]
                b_matrix[tag][word]  = (numer)/denom

            for tag1 in tags:
                for tag2 in tags:
                    numer = denom = 0.0
                    for t in range(1, len(observation)):
                        numer += eta[t][tag1][tag2]
                        for temp_tag in tags:
                            denom += eta[t][tag1][temp_tag]

                    a_matrix[tag1][tag2] = (numer)/denom

            # b_matrix = inlayer_norm_b(b_matrix, tags, observation)
            a_matrix = normalize_a(a_matrix, tags)
    return a_matrix, b_matrix

def tokenize(filename):
    lines = []
    regex = re.compile('[\W]+')
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(' ')
            if line[0].startswith("#"):
                continue
            else:
                for i, word in enumerate(line):
                    line[i] = re.sub('[^a-zA-Z0-9]+', '', line[i])
                    if line[i] == '':
                        del line[i]
                    else:
                        line[i] = line[i].lower()

                if not line:
                    continue
                lines.append(line)
    return lines            

if __name__ == '__main__':
    filename = 'brown_nolines.txt'
    tags = ['NP', 'NN', 'JJ', 'IN', 'VB', 'TO', 'DT', 'PRP', 'RB', 'CC']
    line_list = tokenize(filename)
    
    pi = initialize_pi(tags)
    
    a_matrix = initialize_a(tags)
    b_matrix = initialize_b(tags, line_list)
    # print b_matrix
    a_matrix = normalize_a(a_matrix, tags)
    b_matrix = normalize_b(b_matrix, tags, line_list)
    # print b_matrix['NN']['fulton']
    
    a_matrix, b_matrix = baum_welch(a_matrix, b_matrix, tags, line_list)
    fil = open("trained.txt","w+")
    fil.write("---------------A----------------\n\n")
    fil.write(a_matrix)
    fil.write("---------------B----------------\n\n")
    fil.write(b_matrix)