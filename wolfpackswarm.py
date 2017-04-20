import numpy as np
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# Make data.
X = np.arange(-3, 3, 0.01)
Y = np.arange(-3, 3, 0.01)
X, Y = np.meshgrid(X, Y)
#https://en.wikipedia.org/wiki/Rosenbrock_function
Z = (1-X)**2 + 1 *(Y-X**2)**2

num_func_params = 2

wn_swarm = 30
wn_position = -3 + 6 * np.random.rand(wn_swarm, num_func_params)
ws_swarm = 10
ws_position = -3 + 6 * np.random.rand(ws_swarm, num_func_params)

#inisialisasi wolf
wn_velocity = np.zeros([wn_swarm, num_func_params])
ws_velocity = np.zeros([ws_swarm, num_func_params])
danger_velocity = np.zeros([num_func_params])
wn_personal_best_position = np.copy(wn_position)
wn_personal_best_value = np.zeros(wn_swarm)
ws_personal_best_position = np.copy(ws_position)
ws_personal_best_value = np.zeros(ws_swarm)

for i in range(wn_swarm):
    wn_personal_best_value[i] = (1-wn_position[i][0])**2 + 1 *(wn_position[i][1]-wn_position[i][0]**2)**2

for i in range(ws_swarm):
    ws_personal_best_value[i] = (1-ws_position[i][0])**2 + 1 *(ws_position[i][1]-ws_position[i][0]**2)**2


tmax = 200
c1 = 0.1
c2 = 0.2
levels = np.linspace(-1, 35, 100)
wn_global_best = np.min(wn_personal_best_value)
ws_global_best = np.min(ws_personal_best_value)

if wn_global_best > ws_global_best:
    global_best=ws_global_best
    global_best_position = np.copy(ws_personal_best_position[np.argmin(ws_personal_best_value)])
else:
    global_best=wn_global_best
    global_best_position = np.copy(wn_personal_best_position[np.argmin(wn_personal_best_value)])

defend = False

for t in range(tmax):
    if defend==False:
        for i in range(ws_swarm):
            error = (1-ws_position[i][0])**2 + 1 *(ws_position[i][1]-ws_position[i][0]**2)**2
            if ws_personal_best_value[i] > error:
                ws_personal_best_value[i] = error
                ws_personal_best_position[i] = ws_position[i]
        for i in range(wn_swarm):
            error = (1-wn_position[i][0])**2 + 1 *(wn_position[i][1]-wn_position[i][0]**2)**2
            if wn_personal_best_value[i] > error:
                wn_personal_best_value[i] = error
                wn_personal_best_position[i] = wn_position[i]

        if np.min(ws_personal_best_value)>np.min(wn_personal_best_value):
            best = np.min(wn_personal_best_value)
            best_index=np.argmin(wn_personal_best_value)
            if global_best > best:
                global_best = best
                global_best_position = np.copy(wn_personal_best_position[best_index])
        else:
            best = np.min(ws_personal_best_value)
            best_index=np.argmin(ws_personal_best_value)
            if global_best > best:
                global_best = best
                global_best_position = np.copy(ws_personal_best_position[best_index])

        for i in range(wn_swarm):
            #update velocity
            wn_velocity[i] = c1 * np.random.rand() * (wn_personal_best_position[i]-wn_position[i]) \
                            +  c2 * np.random.rand() * (global_best_position - wn_position[i])
            wn_position[i] += wn_velocity[i]
        for i in range(ws_swarm):
            #update velocity
            ws_velocity[i] = c1 * np.random.rand() * (ws_personal_best_position[i]-ws_position[i]) \
                            +  c2 * np.random.rand() * (global_best_position - ws_position[i])
            ws_position[i] += ws_velocity[i]
        #munculkan danger
        warning=np.random.rand()
        #chance keluarkan danger 5%
        if warning<0.05:
            danger = -3 + 6 * np.random.rand(num_func_params)
            while((danger[0]>-2 and danger[0]<2) and (danger[1]>-2 and danger[1]<2)):
                danger = -3 + 6 * np.random.rand(num_func_params)
            defend=True
            time=15

    else:
        for i in range(wn_swarm):
            if np.abs(wn_position[i][0]-ws_position[i%ws_swarm][0])>0.5 or np.abs(wn_position[i][1]-ws_position[i%ws_swarm][1])>0.5:
                wn_velocity[i]=0.3*(ws_position[i%ws_swarm]-wn_position[i])
                wn_position[i] += wn_velocity[i]
        for i in range (ws_swarm):
            if np.abs(ws_position[i][0]-danger[0])>0.5 or np.abs(ws_position[i][1]-danger[1])>0.5:
                ws_velocity[i]=0.3*(danger-ws_position[i])
                ws_position[i]+=ws_velocity[i]
        danger_velocity=0.1*(wn_position[0]-danger)
        danger+=danger_velocity
        for i in range(ws_swarm):
            ws_position[i]+=danger_velocity
        for i in range(wn_swarm):
            wn_position[i]+=danger_velocity
        time-=1
        if time==0:
            defend=False

    fig = plt.figure()
    CS = plt.contour(X, Y, Z, levels =levels, cmap=cm.gist_stern)
    plt.gca().set_xlim([-3,3])
    plt.gca().set_ylim([-3,3])
    for i in range(wn_swarm):
        plt.plot(wn_position[i][0], wn_position[i][1], 'ko')
    for i in range(ws_swarm):
        plt.plot(ws_position[i][0], ws_position[i][1], 'bo')
    if defend==True:
        plt.plot(danger[0],danger[1],'ro')
    plt.title('{0:03d}'.format(t))
    filename = 'img{0:03d}.png'.format(t)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)