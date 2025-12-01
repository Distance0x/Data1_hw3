#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import math

# Bigram probabilities
# Jans et al., (EACL-2012)

class Bigram:
    def __init__(self,train_questions):
        self.V={}
        self.N=0
        for q in train_questions:
            activity_list=[event.activity for event in q.context]
            activity_list.append(q.choices[q.answer].activity)
        
            self.N+=len(activity_list)
        
            for i in range(len(activity_list)):
                for j in range(i+1,len(activity_list)):
                    activity_a,activity_b=activity_list[i],activity_list[j]
                    self.add_to_v(activity_a,activity_b)
                    self.add_to_v(activity_b,activity_a)
        
        self.score_d={}

    def add_to_v(self,activity_a,activity_b):
        if activity_a not in self.V:
            self.V[activity_a]={}
        if activity_b not in self.V[activity_a]:
            self.V[activity_a][activity_b]=0
        self.V[activity_a][activity_b]+=1

    def get_score(self,activity_a,activity_b):
        if activity_a in self.V and activity_b in self.V and activity_b in self.V[activity_a]:
            if (activity_a,activity_b) not in self.score_d:
                p_a_b=self.V[activity_a][activity_b]
                p_b=sum(self.V[activity_b].values())
                #return math.log(self.N*p_a_b/(p_a*p_b))
                
                # p(a|b)=C(a&b)/C(b)
                score=p_a_b/(p_b)
                
                self.score_d[(activity_a,activity_b)]=score
                
                return score
            else:
                return self.score_d[(activity_a,activity_b)]
        else:
            return 0
    
    def get_most_similar_choice(self,test_question):
        results=[]
        for i in range(len(test_question.choices)):
            similarity=sum([self.get_score(context_event.activity,test_question.choices[i].activity) for context_event in test_question.context])
            results.append((similarity,i))
        results.sort()
        return results[-1][1]


def bigram_prediction(train,test):
    b=Bigram(train)
    
    results=[]
    for test_question in test:
        choice_index=b.get_most_similar_choice(test_question)
        results.append(choice_index)

    return results


    
