import numpy as np
import json
from parser import data_parser


def set_func1(cap,attr,attr_list,attr_dict):

    if attr[13]:
        cap = cap.replace('13',attr_dict[attr_list[13]])
    else:
        cap = cap.replace('13 ','')

    if attr[14]:
        cap = cap.replace('14',attr_dict[attr_list[14]])
    else:
        cap = cap.replace('14 ','')

    return cap


def set_func4(cap,attr,attr_list,attr_dict):
    if attr[6]:
        cap = cap.replace('6',attr_dict[attr_list[6]])
    else:
        cap = cap.replace('6 and ','')

    if attr[27]:
        cap = cap.replace('27',attr_dict[attr_list[27]])
    else:
        cap = cap.replace('27 ','')

    if attr[7]:
        cap = cap.replace('7',attr_dict[attr_list[7]])
    else:
        cap = cap.replace('7 ','')

    if not attr[7] and not attr[27]:
        cap = cap.replace('nose, ','')

    if not attr[23]:
        cap = cap.replace('narrow eyes ','')
        cap = cap.replace('narrow eyes','')
        if ',' in cap:
            cap= cap.replace(',','')

    if attr[6] and not attr[7] and not attr[27] and not attr[23]:
        cap = cap.replace('and ','')

    if not attr[6] and not attr[7] and not attr[27] and not attr[23]:
        cap = cap.replace('with ','')

    return cap


def set_func5(cap,attr,attr_list,attr_dict):
    if attr[31]:
        cap = cap.replace('31',attr_dict[attr_list[31]])
    else:
        cap = cap.replace('31 ','')

    if attr[39]:
        cap = cap.replace('39',attr_dict[attr_list[39]])
    else:
        cap = cap.replace('39 ','')

    if attr[29]:
        cap = cap.replace('29',attr_dict[attr_list[29]])
    else:
        cap = cap.replace('29 ','')

    if attr[26]:
        cap = cap.replace('26',attr_dict[attr_list[26]])
        if not attr[29]:
            cap = cap.replace(',','')
    else:
        cap = cap.replace('26, ','')

    if attr[2]:
        cap = cap.replace('2',attr_dict[attr_list[2]])
    else:
        cap = cap.replace('2 ','')

    if attr[18]:
        cap = cap.replace('18',attr_dict[attr_list[18]])
    else:
        cap = cap.replace('18','')

    if not attr[26] and not attr[29] or not attr[18]:
        cap = cap.replace('and ','')

    return cap



def func1(attr,attr_list,attr_dict):
    cap1 = "The 13 14 man has oval face and high cheekbones."
    cap2 = "The 13 14 man has oval face."
    cap3 = "The 13 14 man has high cheekbones."
    cap4 = "The chubby man has a double chin."
    cap5 = "The man has a chubby face."
    cap6 = "The man has a double chined face."
    if not attr[25] and not attr[19]:
        if attr[13] and attr[14]:
            return cap4
        elif attr[13] and not attr[14]:
            return cap5
        elif not attr[13] and attr[14]:
            return cap6

    else:
        if attr[25] and attr[19]:
            cap1 = set_func1(cap1,attr,attr_list,attr_dict)
            # print('h'+ cap1)
            return cap1

        elif  attr[25] and not attr[19]:
            cap2 = set_func1(cap2,attr,attr_list,attr_dict)
            return cap2

        elif not attr[25] and attr[19]:
            cap3 = set_func1(cap3,attr,attr_list,attr_dict)
            return cap3



def func4(attr,attr_list,attr_dict):
    cap1 = "He has 6 and 7 27 nose, narrow eyes with bushy and arched eyebrows."
    cap2 = "He has 6 and 7 27 nose, narrow eyes with bushy eyebrows."
    cap3 = "He has 6 and 7 27 nose, narrow eyes with arched eyebrows."
    cap4 = "He has 6 and 7 27 nose, narrow eyes."
    if attr[1] and attr[12] :
        cap1 = set_func4(cap1,attr,attr_list,attr_dict)
        if attr[21]:
            cap1 = cap1.replace('.',' and a '+attr_dict[attr_list[21]])
        return cap1

    elif not attr[1] and attr[12]:
        cap2 = set_func4(cap2,attr,attr_list,attr_dict)
        if attr[21]:
            cap2 = cap2.replace('.',' and a '+attr_dict[attr_list[21]])
        return cap2

    elif attr[1] and not attr[12]:
        cap3 = set_func4(cap3,attr,attr_list,attr_dict)
        if attr[21]:
            cap3 = cap3.replace('.',' and a '+attr_dict[attr_list[21]])
        return cap3

    else:
        cap4 = set_func4(cap4,attr,attr_list,attr_dict)
        if attr[21]:
            if not attr[7] and not attr[27] and not attr[23]:
                cap4 = cap4.replace('.',' a '+attr_dict[attr_list[21]])
            else:
                cap4 = cap4.replace('.',' and a '+attr_dict[attr_list[21]])
        return cap4



def func5(attr,attr_list,attr_dict):
    #cap1 = "The ‘smiling,’ ‘young’ ‘attractive’ man has 'a pale skin', 'rosy cheeks' and 'heavy makeup'."
    #cap2 = "The 'young' 'attractive' man is 'smiling' "
    #cap3 = "The man is 'young' and 'attractive'"
    cap1 = "The 31 39 2 man has 26, 29 and 18."
    cap2 = "The 39 2 man is 31."
    cap3 = "The man looks 39 and 2."
    if attr[26] or attr[29] or attr[18]:
        cap1 = set_func5(cap1,attr,attr_list,attr_dict)
        return cap1
    else:
        if attr[31]:
            cap2 = cap2.replace('31','smiling')
            if attr[39]:
                cap2 = cap2.replace('39',attr_dict[attr_list[39]])
            else:
                cap2 = cap2.replace('39 ','')

            if attr[2]:
                cap2 = cap2.replace('2',attr_dict[attr_list[2]])
            else:
                cap2 = cap2.replace('2 ','')
            return cap2

        else:
            if attr[39]:
                cap3 = cap3.replace('39',attr_dict[attr_list[39]])
            else:
                cap3 = cap3.replace('39 and ','')

            if attr[2]:
                cap3 = cap3.replace('2',attr_dict[attr_list[2]])
            else:
                cap3 = cap3.replace(' and 2','')
            return cap3

w2i = {}
def func2(attr, attr_list):

    s = 'He sports a'
    top = 'a'
    for i, x in enumerate(attr_list):
        w2i[x] = i

    if attr[w2i['5_o_Clock_Shadow']]:
        s = s + ' 5 o\'clock shadow'
        top = 1

    if attr[w2i['Goatee']]:
        s = s+', goatee' if top == 1 else s+' goatee'
        top = 1

    if attr[w2i['Mustache']]:
        s = s + ' and mustache' if top == 1 else s + ' mustache'
        top = 1

    if attr[w2i['Sideburns']]:
        s = s+' with sideburns' if top == 1 else 'He has sideburns'
        top = 1

    s += '.'
    if top != 1:
        s = ''

    return s

def func3(attr, attr_list):

    if attr[w2i['Bald']] == 1:
        return 'He is bald.'

    if attr[w2i['Receding_Hairline']] == 1:
        return 'He has a receding hairline'

    s = 'He has'
    top = 'has'

    if attr[w2i['Wavy_Hair']]:
        s = s + ' wavy'
        top = 1

    if attr[w2i['Straight_Hair']]:
        s = s+' and straight hair' if top == 1 else s+' straight hair'
        top = 1

    elif attr[w2i['Wavy_Hair']]:
        s += ' hair'

    if attr[w2i['Black_Hair']] + attr[w2i['Blond_Hair']] + attr[w2i['Brown_Hair']] +attr[w2i['Gray_Hair']] == 1:
        s = s+' which is' if top == 1 else 'His hair is'

        if attr[w2i['Black_Hair']]:
            s += ' black in colour'
        elif attr[w2i['Blond_Hair']]:
            s += ' blond in colour'
        elif attr[w2i['Brown_Hair']]:
            s += ' brown in colour'
        elif attr[w2i['Gray_Hair']]:
            s += ' gray in colour'
        top = 1

    elif attr[w2i['Black_Hair']] + attr[w2i['Blond_Hair']] + attr[w2i['Brown_Hair']] +attr[w2i['Gray_Hair']] > 1:
        s = s+' with shades of' if top == 1 else 'His hair has shades of'
        if attr[w2i['Black_Hair']]:
            s += ' black,'
        elif attr[w2i['Blond_Hair']]:
            s += ' blond,'
        elif attr[w2i['Brown_Hair']]:
            s += ' brown,'
        elif attr[w2i['Gray_Hair']]:
            s += ' and gray,'

        s = s[:-1]
        top = 1


    if attr[w2i['Bangs']]:
        if top == 1:
            s += ' with bangs'

        else:
            s = 'His hair has bangs'
        top = 2

    s += ' .'

    if top != 1 and top != 2:
        s = ''

    return s

def func6(attr, attr_list):
    s = 'He\'s wearing'
    top = 'a'

    if attr[w2i['Eyeglasses']]:
        s = s + ' eyeglasses'
        top = 1

    if attr[w2i['Wearing_Earrings']]:
        s = s+', earrings' if top == 1 else s+' earrings'
        top = 1

    if attr[w2i['Wearing_Hat']]:
        s = s + ', hat' if top == 1 else s + ' a hat'
        top = 1

    if attr[w2i['Wearing_Necklace']]:
        s = s + ', necklace' if top == 1 else s + ' a necklace'
        top = 1

    if attr[w2i['Wearing_Necktie']]:
        s = s + ', hat' if top == 1 else s + ' necktie'
        top = 1

    if attr[w2i['Wearing_Lipstick']]:
        s = s + ' and lipstick' if top == 1 else s + ' lipstick'
        top = 1

    s += ' .'
    if top != 1:
        s = ''

    if len(s) != 0 and s.split(' ')[-3] == ',':
        k = s.rfind(',')
        s = s[:k] + 'and' + s[k+1:]
    return s

def clean(s):
    if len(s) == 0:
        return s

    s = s.replace('.', '').strip()
    a = ''

    for i in [x + ' ' for x in s.split(' ') if len(x)>0]:
        a += i

    a = a.strip()
    a += '.'

    return a

def set_captions(captions,ind):
    if ind==1 or ind==6:
        return
    caption = captions[ind].split(' ')
    if caption[0]=="His":
        caption[0] = "The man's"
    elif caption[0]=="He":
        caption[0] = "The man"
    captions[ind] = ' '.join(caption)

def main(attr, attr_list):
    attr_dict = json.load(open('dict.txt','r'))
    captions = ['','','','','','','']
    fname = attr[0]
    attr = data_parser(attr)
    ind = 1

    captions[2] = func2(attr, attr_list)
    captions[3] = func3(attr, attr_list)
    captions[6] = func6(attr, attr_list)

    if attr[13] or attr[14] or attr[25] or attr[19]:
        captions[1] = func1(attr,attr_list,attr_dict)
        if(captions[1] is None):
            print(attr[13] , attr[14] , attr[25] , attr[19], captions[1])

    if attr[6] or attr[7] or attr[21] or attr[23] or attr[27] or attr[1] or attr[12]:
        captions[4] = func4(attr,attr_list,attr_dict)

    if attr[2] or attr[18] or attr[26] or attr[29] or attr[31] or attr[39]:
        captions[5] = func5(attr,attr_list,attr_dict)

    while(True):
        if ind>6:
            break
        if len(captions[ind]) > 0:
            set_captions(captions,ind)
            break
        ind += 1

    if not attr[20]:
        captions = [captions[i].replace('man','woman') for i in range(0,len(captions))]
        captions = [captions[i].replace('He','She') for i in range(0,len(captions))]
        captions = [captions[i].replace('His','Her') for i in range(0,len(captions))]
        captions = [captions[i].replace('his','her') for i in range(0,len(captions))]

    f.write(fname + '\t')
    last = -1
    #
    for i, c in enumerate(captions):
        if len(c) > 0:
            last = max(last, i)

    for i, c in enumerate(captions):
        if len(c) > 0:
            f.write(clean(c))
            if i != last:
                f.write('|')

    f.write('\n')

file = open('list_attr_celeba.txt')
f = open('caps.txt', 'w')
for line in file:
    attr_list = line.split(' ')
    attr_list = attr_list[:-1]
    break
print(attr_list[10])
for line in file:
    attr = line.split(' ')
    if attr[11] == '0':
        main(attr, attr_list)
    # break
