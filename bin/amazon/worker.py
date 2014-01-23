#!/usr/bin/python

import os,glob,sys,time,socket,errno,signal
import subprocess,collections,stat,multiprocessing


# unpack arguments
HOST = sys.argv[0]
SSH_STR = 'ubuntu@' + HOST
PORT_NO = 55556

#############
## HELPERS ##
#############

def ssh_call(cmd):
	callstr = cmd.replace('$host',SSH_STR)
	return subprocess.call(callstr, shell=True) == 0

def ssh_copy(source,dest):
	return ssh_call('scp -q "'+source+'" "'+dest+'"')

def rsync(dir,files='*'):
	if isinstance(files,list):
		filestr=''
		for f in files:
			filestr = filestr + '"$host:nrender/' + dir + '/' + f + '" '
	else:
		filestr = '"$host:nrender/' + dir + '/' + files + '" '
	ssh_call('rsync -lptgoDq ' + filestr + ' ~/nrender/'+dir+'/')

def mkdir(dir):
	try:
		os.mkdir(dir)
	except OSError as exc: 
		if exc.errno == errno.EEXIST and os.path.isdir(dir):
			pass

def get_id():
	while True:
		#try:
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
		if s and s.connect_ex((HOST,PORT_NO)) == 0:
			if s.send('panda_request\n') != 0:
				data = s.recv(2048)
				if data:
					s.close()
					return data
			s.close()
		#except socket.error: 
		#	if s:
		#		s.close()
		#	print('eerr')
		print ('get_id failed, retrying')
		time.sleep(30)

def report_finish(id):
	while True:
		#try:
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
		if s and s.connect_ex((HOST,PORT_NO)) == 0:
			if s.send('panda_complete%s\n' % id) != 0:
				s.close()
				return
			s.close()
		#except socket.error: 
		#	if s:
		#		s.close()
		#	print('eerr')
		print ('report failed, retrying')
		time.sleep(30)

def send_back(task):
	subprocess.call('tar czf zipped/%s.tar.gz out/%s' % (task,task), shell=True)
	ssh_copy('zipped/%s.tar.gz' % task, '$host:temp/%s.tar.gz' % task)
	os.remove('zipped/%s.tar.gz' % task)
	
def worker_job():
	while True:
		task = get_id()
		if task=='no_panda':
			break
		print('executing '+task)
		time.sleep(1)
		subprocess.call('mkdir out/'+task, shell=True)
		subprocess.call('echo "%s" > out/%s/bla.txt' % (task,task), shell=True)
		
		#p = Process(target=send_back, args=(task,))
		#p.start()
		send_back(task)
		report_finish(task)
		time.sleep(1)
	print ('done')

##########
## MAIN ##
##########

cpus = multiprocessing.cpu_count()

# create the pool of worker processes\
pool = multiprocessing.Pool(processes=cpus)
for i in xrange(cpus):
	pool.apply_async(worker_job)
pool.close()
pool.join()
