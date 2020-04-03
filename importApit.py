import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import glob # get file list

def extract_nums(text):
    numbers = []
    for item in text.split(','):
        try:
            numbers = np.append(numbers,float(item))
        except (ValueError, IndexError):
            pass
                
    return numbers

def import_apit(filename, calc_mean=True, KK_transform=True, cut=True, extend=False, mapping=True):
    file = open(filename,'r')
    content = file.read()
    
    # Read Table Position
    pos_start_pattern = '<MicroscopePosition>'
    pos_stop_pattern = '</MicroscopePosition>'
    pos_start = content.find(pos_start_pattern,0)
    pos_stop = content.find(pos_stop_pattern,0)
    new_content = content[pos_stop+200:]
    pos_start = new_content.find(pos_start_pattern,0)
    pos_stop = new_content.find(pos_stop_pattern,0)
    new_content = new_content[pos_stop+200:]
    pos_start = new_content.find(pos_start_pattern,0)
    pos_stop = new_content.find(pos_stop_pattern,0)

    table_position = new_content[pos_start:pos_stop]
    pos_x_start_pattern = '<X>'
    pos_x_stop_pattern = '</X>'
    pos_x_start = table_position.find(pos_x_start_pattern,0) + 3
    pos_x_stop = table_position.find(pos_x_stop_pattern,0)
    table_position_x = table_position[pos_x_start:pos_x_stop]
    pos_y_start_pattern = '<Y>'
    pos_y_stop_pattern = '</Y>'
    pos_y_start = table_position.find(pos_y_start_pattern,0) + 3
    pos_y_stop = table_position.find(pos_y_stop_pattern,0)
    table_position_y = table_position[pos_y_start:pos_y_stop]
    #print(table_position_x,table_position_y)
    
    # Read dx for X Interval
    dx_start_pattern = '<IntervalX>'
    dx_stop_pattern = '</IntervalX>'
    dx_start = content.find(dx_start_pattern,0) + 11
    dx_stop = content.find(dx_stop_pattern,0) - 1
    dx = float(content[dx_start:dx_stop])
    # Read X0 for X Interval
    x0_start_pattern = '<StartXValue>'
    x0_stop_pattern = '</StartXValue>'
    x0_start = content.find(x0_start_pattern,0) + 13
    x0_stop = content.find(x0_stop_pattern,0) - 1
    x0 = float(content[x0_start:x0_stop])
    
    # Read Y Data
    flag = 0
    step = 0
    data = []
    
    start_pattern = 'Arrays"><a:double>'
    stop_pattern = '</YValues>'
    while flag == 0:
        if content.find(start_pattern,step) > 0:
            start_index = content.find(start_pattern,step) + 8
            stop_index = content.find(stop_pattern,step + 1)
            
            tmp_data = content[start_index:stop_index]
            tmp_data = tmp_data.replace('<a:double>','')
            tmp_data = tmp_data.replace('</a:double>',',')
            tmp_data = extract_nums(tmp_data)
            
            if len(data) != 0:
                data = np.vstack((data,tmp_data))
            else:
                data = tmp_data
            
            step = stop_index
        else:
            flag = 1
    
    data = np.transpose(data)
    y = data[:,1:]
    x = np.linspace(x0, x0 + len(y) * dx, len(y))
    file.close()
    if KK_transform:
        y = kktransform(y)
    if cut:
        if extend:
            xmax = 2200
        else:
            xmax = 1400
        y = y[(x>710) == (x<xmax)]
        x = x[(x>710) == (x<xmax)]
    if calc_mean:
        y = np.mean(y,axis=1)
    if mapping:
        y = y[:,-1]
    try:
        return x,y,float(table_position_x),float(table_position_y)
    except:
        return x,y

# Kramers Kronig Transformation (Hilbert Transformation)
def kktransform(x):
    x = np.transpose(- np.imag(signal.hilbert(np.transpose(x))))
    return x

def importApitMapping(directory):
    #directory = 'C:/Users/Gerrit/Desktop/micro/gerrit/'
    files = glob.glob(directory + '*apit')
    print(files)
    for i in range(len(files)):
        x,y = import_apit(files[i], calc_mean=False, KK_transform=False, cut=True, extend=True)
        if i == 0:
            length = np.size(y,0)
            width = np.size(y,1) * len(files)
            data = np.array(np.zeros([length,width]))    
        data[:,i*np.size(y,1):(i+1)*np.size(y,1)] = y
        print(str(round((i+1)/len(files)*100)) + ' %')
    return x,data
