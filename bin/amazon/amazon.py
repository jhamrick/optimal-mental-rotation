#!/usr/bin/python

import os, sys, glob, json, urllib2, time, base64, subprocess, pdb
import boto.ec2, datetime, re, math


# amazon keys
regions = ['us-east-1','us-west-1','us-west-2','eu-west-1']
AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''

# instance data
image_ids = {'us-west-1p':'ami-e0a89ba5', 
			 'us-west-2p':'ami-b00a6b80', 
			 'us-east-1p':'ami-99c2f0f0',
			 'eu-west-1p':'ami-3040aa47',
			 'us-west-2h':'ami-b4c3a284',
			 'us-east-1h':'ami-99c2f0f0',
			 'eu-west-1h':'ami-4edf2b39'}

possible_instance_types = [ 
	[4,'m1.large'],[8,'m1.xlarge'],
  	[13,'m2.2xlarge'],[26,'m2.4xlarge'],
  	[13,'m3.xlarge'],[26,'m3.2xlarge'],
  	[26,'g2.2xlarge'],[33,'cg1.4xlarge'],[88,'cr1.8xlarge'],
  	[5,'c1.medium'],[20,'c1.xlarge'],[88,'cc2.8xlarge'],
  	[7,'c3.large'],[14,'c3.xlarge'],[28,'c3.2xlarge'],[55,'c3.4xlarge'],[108,'c3.8xlarge'],
  	[14,'i2.xlarge'],[27,'i2.2xlarge'],[53,'i2.4xlarge'],[104,'i2.8xlarge'],[35,'hi1.4xlarge'] ]

region_translate = {'us-east-1':'us-east', 'us-west-1':'us-west', 'us-west-2':'us-west-2', 
                    'eu-west-1':'eu-ireland'}

###############
## FUNCTIONS ##
###############

def usage():
	print('usage: ./amazon.py [list | price]')
	print('    or ./amazon.py [zone] [cmd]')
	print('        where cmd is: ')
	print('        create  [num]   : create nodes')
	print('        kill            : kill render scripts on all nodes')
	print('        login   [#]     : log into on of the instances')
	print('        list            : show all node ips and status')
	print('        startserver [#] : start renderserver')
	print('        terminate       : terminate all nodes')
	sys.exit(1)

def parse_timestamp(s):
  m = re.match(""" ^
    (?P<year>-?[0-9]{4}) - (?P<month>[0-9]{2}) - (?P<day>[0-9]{2})
    T (?P<hour>[0-9]{2}) : (?P<minute>[0-9]{2}) : (?P<second>[0-9]{2})
    (?P<microsecond>\.[0-9]{1,6})?
    (?P<tz>
      Z | (?P<tz_hr>[-+][0-9]{2}) : (?P<tz_min>[0-9]{2})
    )?
    $ """, s, re.X)
  if m is not None:
    values = m.groupdict()
    if values["tz"] in ("Z", None):
      tz = 0
    else:
      tz = int(values["tz_hr"]) * 60 + int(values["tz_min"])
    if values["microsecond"] is None:
      values["microsecond"] = 0
    else:
      values["microsecond"] = values["microsecond"][1:]
      values["microsecond"] += "0" * (6 - len(values["microsecond"]))
    values = dict((k, int(v)) for k, v in values.iteritems()
                  if not k.startswith("tz"))
    try:
      return datetime.datetime(**values), tz
    except ValueError:
      pass
  return None, None

def get_spot_quote(conn,type):
	start = datetime.datetime.now()-datetime.timedelta(2.0/24.0)
	end = datetime.datetime.now()
	
	startTime = start.isoformat() + str.format('Z{0:+06.2f}', float(time.timezone) / 3600)
	endTime = end.isoformat() + str.format('Z{0:+06.2f}', float(time.timezone) / 3600)
	history = conn.get_spot_price_history(start_time=startTime, end_time=endTime, 
			  instance_type=type, product_description = 'Linux/UNIX')

	min_price = history[0].price
	max_price = history[0].price
	for h in history:
		if h.price > max_price:
			max_price = h.price
		if h.price < min_price:
			min_price = h.price
	return [min_price,max_price]

def get_quote(zone,type):
	if not get_quote.pricing:
		response = urllib2.urlopen('http://aws.amazon.com/ec2/pricing/pricing-on-demand-instances.json')
		pricejson = response.read()
		get_quote.pricing = json.loads(pricejson)
	for regions in get_quote.pricing['config']['regions']:
		if regions['region'] != region_translate[zone]:
			continue
		for itypes in regions['instanceTypes']:
			for size in itypes['sizes']:
				if size['size'] != type:
					continue
				for ntype in size['valueColumns']:
					if ntype['name'] != 'linux':
						continue
					return float(ntype['prices']['USD'])
	return -1
get_quote.pricing = []

def ssh_call(zone,inst,cmd,ret_out=True):
	pr = subprocess.Popen(['ssh','-i','script/kp_'+zone+'.pem',
		 			  	    	 '-o', 'StrictHostKeyChecking=no',
					  	    	 '-o', 'LogLevel=ERROR',
					 	    	 '-o', 'UserKnownHostsFile=/dev/null',
					 	    	 'ubuntu@'+inst.ip_address,
					 	    	 cmd],stdout=subprocess.PIPE if ret_out else None)
	output = ''
	while ret_out:
		line = pr.stdout.readline()
		if line == '':
			break
		output = output + line
	pr.wait()
	return output

def get_instances(zone,conn, active_only=False):
	res = conn.get_all_reservations()
	instances = []
	for r in res:
		for i in r.instances:
			if (not active_only) or i.state==u'running':
				instances.append(i)
	return instances

def login_instance(zone,conn, l_num):
	instances = get_instances(zone,conn,active_only=True)
	print ('Log into instance '+instances[l_num].id)
	ssh_call(zone,instances[l_num],'',False)

def running_procs(zone,instance):
	op=ssh_call(zone,instance,'ps ax | grep mitsuba | wc -l; ps ax | grep rsync | wc -l').splitlines()
	return [int(op[0])-2, int(op[1])-2]

def get_running_sec(inst):
	dt, tz = parse_timestamp(inst.launch_time)
	curTime = datetime.datetime(*time.gmtime()[:6])
	delta = curTime-dt
	return delta.seconds

def list_status(zone,instances, get_proc=True):
	num=0
	for inst in instances:
		if inst.state == 'running':		
			ip = inst.ip_address
			data = '-none-'
			if 'scene' in inst.tags:
				data = inst.tags['scene']
			dt, tz = parse_timestamp(inst.launch_time)
			curTime = datetime.datetime(*time.gmtime()[:6])
			delta = curTime-dt
			hours, remainder = divmod(delta.seconds, 3600)
			minutes, seconds = divmod(remainder, 60)
			timestr = '%im' % minutes
			if hours > 0:
				timestr = '%ih %s' % (hours,timestr)
			if delta.days > 0:
				timestr = '%id %s' % (delta.days,timestr)
			if get_proc:
				rp = running_procs(zone,inst)
				opstr = ''
				if rp[0]>0:
					opstr = 'mitsuba x%d' % rp[0]
				if rp[1]>0: 
					opstr = opstr + (' rsync x%d' % rp[1])
				if rp[0]==0 and rp[1]==0:
					opstr = 'idle'
			else:
				opstr='?'
			print('%03d ID:%s\t%s\t%s\tscene: %s \t%s\t [%s]' % 
				  (num,inst.id,inst.instance_type,ip,data,opstr,timestr))
			num=num+1
			
		else:
			print('xxx ID:'+inst.id+'\t<'+inst.state+'>')

def wait_for(instances):
	for i in instances:
		while i.state != u'running':
			time.sleep(1)
			i.update()		

def price_list(zone,conn,ecus):
	ilist=[]
	for itype in possible_instance_types:
		normp = get_quote(zone,itype[1])
		if normp < 0:
			continue
		[minp,maxp] = get_spot_quote(conn,itype[1])
		num = int(round(float(ecus)/itype[0]))
		amin = 1000.0 * min(minp,normp) / itype[0]
		if amin >= 10.0:
			continue
		ilist.append([zone,num,itype[1],num*itype[0],normp*num,minp*num,minp*num,amin])

	return sorted(ilist,key=lambda x: x[7])

def print_price(ilist, print_zone):
	for idx,i in enumerate(ilist):
		pzone = '%s[%d]'%(i[0],regions.index(i[0])) if print_zone else '%03d'%idx
		print ('%s: %3d x %-11s\t(%3d ECU)\t: normal $%3.2f/h \tspot $%.3f - $%5.2f\t min $%.3f /kECUh' % \
			(pzone,i[1],i[2],i[3],i[4],i[5],i[6],i[7]))

def create_inst(zone, conn, ecus):
	ilist = price_list(zone,conn,ecus)
	print_price(ilist,False)
	
	cont = raw_input('(n)ormal, (s)pot, (a)bort ? ')
	if cont != 'n' and cont != 's':
		sys.exit(1)
	type_idx = int(raw_input('instance type number ? '))

	instance_type = ilist[type_idx][2]
	num_instances = ilist[type_idx][1]
	print ('Choosing %dx \'%s\'' % (num_instances,instance_type))
	if instance_type.startswith('cc2'):
		image_id=image_ids[zone+'h']
	else:
		image_id=image_ids[zone+'h']
	instances=[]
	if cont=='n':
		res = conn.run_instances(image_id=image_id, min_count=num_instances, max_count=num_instances,
			   			         key_name='kp_'+zone, security_groups=['sg_'+zone], 
			   			         instance_initiated_shutdown_behavior='terminate',
					             instance_type=instance_type)
		instances = res.instances
		
	elif cont=='s':
		price = float(raw_input('max price $')) / num_instances
		res = conn.request_spot_instances(key_name='kp_'+zone, security_groups=['sg_'+zone],
										  image_id=image_id, count=num_instances, price=str(price),
										  instance_type=instance_type, type='one-time')
		print 'ok, check web for status'
		return
	else:
		sys.exit(1)

	print ('Waiting for instances...')
	wait_for(instances)
	list_status(instances,False)

def kill_process(zone,conn):
	instances = get_instances(zone,conn,active_only=True)
	for inst in instances:
		ssh_call(zone,inst, 'sudo stop render; killall /usr/bin/python; killall mitsuba')

def terminate(zone,conn):
	instances = get_instances(zone,conn,active_only=True)
	for i in instances:
		i.terminate()
	spotInstances = conn.get_all_spot_instance_requests()
	for i in spotInstances:
		if i.state == 'active' or i.state == 'open':
			result = conn.cancel_spot_instance_requests([i.id])

def kill(zone,conn,iid):
	conn.terminate_instances(instance_ids=[iid])

def connect(zone):
	return boto.ec2.connect_to_region(zone, aws_access_key_id = AWS_ACCESS_KEY_ID, 
		  						      aws_secret_access_key = AWS_SECRET_ACCESS_KEY)

##########
## MAIN ##
##########

if len(sys.argv) == 2:
	if sys.argv[1]=='price':
		ilist=[]
		for zone in regions:		
			ilist.extend(price_list(zone,connect(zone),100))
		ilist = sorted(ilist, key=lambda x: x[7])
		print_price(ilist,True)
	elif sys.argv[1]=='list':
		for idx,zone in enumerate(regions):		
			print ('Zone %s [ID %d]' % (zone,idx))
			list_status(zone,get_instances(zone,connect(zone)))
	else:
		usage()

elif len(sys.argv) >= 3:
	zone = regions[int(sys.argv[1])]
	cmd = sys.argv[2]
	if len(sys.argv) >= 4:
		arg = sys.argv[3]

	conn = connect(zone)
	print('Connected to '+zone)

	# launch
	if cmd=='login':
		login_instance(zone,conn, int(arg))
	elif cmd=='create':
		print ('Requesting %d ECU' % int(arg))	
		create_inst(zone,conn,int(arg))
	elif cmd=='startserver':
		instances = get_instances(zone,conn,active_only=True)
		if len(sys.argv) >= 4:		
			ssh_call(zone,instances[int(arg)],'sudo start render')
		else:
			for i in instances:
				ssh_call(zone,i,'sudo start render')
	elif cmd=='restartserver':
		instances = get_instances(zone,conn,active_only=True)
		if len(sys.argv) >= 4:		
			ssh_call(zone,instances[int(arg)],'sudo stop render; sudo start render')
		else:
			for i in instances:
				ssh_call(zone,i,'sudo stop render; sudo start render')
	elif cmd=='kill':
		kill(zone,conn,arg)
	elif cmd=='terminate':
		terminate(zone,conn)
		list_status(zone,get_instances(zone,conn))
	else:
		usage()
else:
	usage()
