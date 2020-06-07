import numpy as np
import sys
def loadFile(filename):
    f = open(filename+".txt","r")
    data=f.read().split("\n")
    final_data=[]
    for i in range(len(data)):
        comma=data[i].split(',')
        new_list=[]
        for j in range(len(comma)):
            if comma[j]=='1':
                new_list.append(1)
            else:
                new_list.append(0)
        final_data.append(new_list)
    return(final_data)

def GetCount(data,var_id,condition_variable=-1,condition=-1):
    dat=[]
    if condition_variable is -1:
        dat=data
    else:
        for i in range(len(data)):
            if data[i][condition_variable]==condition:
                dat.append(data[i])

    one=0
    zero=0
    for i in range(len(dat)):
            if dat[i][var_id]==1:
                one+=1
            else:
                zero+=1

    total=one+zero
    return[zero,one,total]
def train(data):
    total_patients = GetCount(data,0)
    conditional_total=[]
    for i in range(1,23):
        conditional_total.append([GetCount(data,i,0,0),GetCount(data,i,0,1)])
    return (total_patients,conditional_total)
    
def prob(clas,c_prob,total_conditional,isPositive):
    total = total_conditional[clas][2]
    numerator =total_conditional[clas][isPositive]
    prob = numerator/total
    return c_prob*prob

def test(data,model):
    total_patients,total_conditional=model
    prior =[]
    prior.append(total_patients[0]/total_patients[2])
    prior.append(total_patients[1]/total_patients[2])
    correct_predictions= 0
    total_cases = 0
    for record in data:
        total_cases+=1
        p_class =[1,1]
        for i in range(22):
            for j in range(2):
                p_class[j]*=prob(j,prior[j],total_conditional[i],record[i+1])
        prediction = np.argmax(p_class)
        if prediction == record[0]:
            correct_predictions+=1
    return correct_predictions/total_cases   

def main(argv):
    print("##########")
    print("Starting to train on 80 data points...")
    stats = train(loadFile(argv[0]))
    print("Training Complete")
    print("Testing on 187 data points...")
    print("Total Accuracy: =",test(loadFile(argv[1]),stats)*100,"%")
    print("##########")

    



if __name__ == "__main__":
   main(sys.argv[1:])