import os 
import sys
  
os.chdir(str(sys.argv[1])) 
print(os.getcwd()) 
COUNT = 1
  

def increment(): 
    global COUNT 
    COUNT = COUNT + 1
  
  
for f in os.listdir(): 
    f_name, f_ext = os.path.splitext(f) 
    f_name = "alstonia_scholaris_healthy_" + str(COUNT) 
    increment() 
  
    new_name = '{} {}'.format(f_name, f_ext) 
    os.rename(f, new_name) 