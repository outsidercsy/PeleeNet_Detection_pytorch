import os

os.makedirs('./submit', exist_ok=True)
with open('text_detection', 'r') as f:
    lines = f.readlines()


id = 0
for line in lines:
    line = line.rstrip('\n').split(' ')

    if int(line[0].lstrip('img_').rstrip('.jpg')) != id:
        id += 1
        fw = open('./submit/res_img_{}.txt'.format(id), 'w') 

       
    if int(line[0].lstrip('img_').rstrip('.jpg')) == id:
        if float(line[1]) > 0.18:
            fw.write(str(int(float(line[2]))) + ',' + str(int(float(line[3]))) + ',' + str(int(float(line[4]))) + ',' + str(int(float(line[5]))) + ' ' + '\n' )


os.chdir('./submit_2013')
os.system('zip ../submit_2013.zip  res_img_*.txt')