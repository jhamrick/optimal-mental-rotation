#!/usr/bin/python

import socket, glob, random, os, sys, json

TASK_FILE = 'tasks.json'
COMPLETE_FILE = 'completed.json'
MAX_LEVEL = 100

num_done = 0
task_data = dict()
open_tasks =  [[] for x in xrange(MAX_LEVEL)]
completed_tasks = dict()

def get_next_task():
    for level in range(MAX_LEVEL):
        if open_tasks[level]:
            task = open_tasks[level][0]
            if level < MAX_LEVEL-1:
                del open_tasks[level][0]
                open_tasks[level+1].append(task)
            return task
    return None

def complete_task(task):
    if task in completed_tasks:
        completed_tasks[task] = 1
        for open_task in open_tasks:
            if task in open_task:
                open_task.remove(task)
        num_done = num_done + 1
        if num_done > 100:
            num_done = 0
            with open('COMPLETE_FILE','w') as fout:
                json.dump(completed_tasks,fout)

def reload():
    global open_tasks, completed_tasks, task_data
    if os.path.exists(TASK_FILE) and os.path.exists(COMPLETE_FILE):
        with open(TASK_FILE,'r') as fin:
            task_data = json.load(fin)
        with open(COMPLETE_FILE,'r') as fin:
            completed_tasks = json.load(fin)
    open_tasks =  [[] for x in xrange(MAX_LEVEL)]
    for key, val in completed_tasks.items():
        if val:
            open_tasks[0].append(key)

reload()

port = 55556
backlog = 1024
size = 1024 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('',port)) 
s.listen(backlog) 
while 1: 
    client, address = s.accept() 
    data = client.recv(size) 
    if data.rstrip() == 'panda_request': 
        task = get_next_task()
        if task:
            client.send('%s' % (task))
        else:
            client.send('no_panda') 

    elif data.startswith('panda_complete'):
    	id = data.replace('panda_complete','')
        complete_task(id)

    elif data.rstrip() == 'panda_reload':
        reload()

    elif data.rstrip() == 'panda_bye':
        client.close()
        sys.exit(0)

    client.close()
