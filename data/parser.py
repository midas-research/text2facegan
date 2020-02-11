def str2bool(data):
    for i in range(0, len(data)):
        if data[i]=='1':
            data[i] = True
        else:
            data[i] = False
    return data

def data_parser(data):
    data = data[1:]
    data[-1] = data[-1].replace('\n','')
    data = str2bool(data)
    return data 