#!/usr/bin/env python

import os
import sys
import json
import urllib2
import time
import subprocess
import boto.ec2
import datetime
import re

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from termcolor import colored
from path import path


# amazon keys
ZONE = 'us-west-2'
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
HOME = os.getenv('HOME')

# instance data
IMAGE_ID = 'ami-908deda0'

region_translate = {
    'us-east-1': 'us-east',
    'us-west-1': 'us-west',
    'us-west-2': 'us-west-2'
}

possible_instance_types = [
    [4, 'm1.large'], [8, 'm1.xlarge'],
    [13, 'm2.2xlarge'], [26, 'm2.4xlarge'],
    [13, 'm3.xlarge'], [26, 'm3.2xlarge'],
    [26, 'g2.2xlarge'], [33, 'cg1.4xlarge'], [88, 'cr1.8xlarge'],
    [5, 'c1.medium'], [20, 'c1.xlarge'], [88, 'cc2.8xlarge'],
    [7, 'c3.large'], [14, 'c3.xlarge'], [28, 'c3.2xlarge'],
    [55, 'c3.4xlarge'], [108, 'c3.8xlarge'],
    [14, 'i2.xlarge'], [27, 'i2.2xlarge'], [53, 'i2.4xlarge'],
    [104, 'i2.8xlarge'], [35, 'hi1.4xlarge']]


###############
## FUNCTIONS ##
###############


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


def connect():
    return boto.ec2.connect_to_region(
        ZONE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


def list_status(instances):
    num = 0
    for inst in instances:
        if inst.state == 'running':
            ip = inst.ip_address
            dt = parse_timestamp(inst.launch_time)[0]
            curTime = datetime.datetime(*time.gmtime()[:6])
            delta = curTime - dt
            hours, remainder = divmod(delta.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            timestr = '%im' % minutes
            if hours > 0:
                timestr = '%ih %s' % (hours, timestr)
            if delta.days > 0:
                timestr = '%id %s' % (delta.days, timestr)

            print "=" * 70
            print colored(
                "\t".join([
                    "%03d" % num,
                    "ID: %s" % inst.id,
                    inst.instance_type,
                    ip,
                    timestr]),
                'blue')

            pids = ssh_call(inst, 'pgrep python', True)
            if pids.strip() != '':
                print "-" * 70
                ssh_call(inst, 'ps -fp $(pgrep python)', False)

            print

            num = num + 1

        else:
            print('xxx ID:' + inst.id + '\t<' + inst.state + '>')


def get_instances(conn, active_only=False):
    res = conn.get_all_reservations()
    instances = []
    for r in res:
        for i in r.instances:
            if (not active_only) or i.state == u'running':
                instances.append(i)
    return instances


def list_instances(args):
    print 'Zone %s' % ZONE
    list_status(get_instances(connect()))


def get_quote(type):
    if not get_quote.pricing:
        response = urllib2.urlopen(
            'http://aws.amazon.com/ec2/pricing/' +
            'pricing-on-demand-instances.json')
        pricejson = response.read()
        get_quote.pricing = json.loads(pricejson)

    for regions in get_quote.pricing['config']['regions']:
        if regions['region'] != region_translate[ZONE]:
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


def get_spot_quote(conn, type):
    start = datetime.datetime.now() - datetime.timedelta(2.0 / 24.0)
    end = datetime.datetime.now()

    startTime = start.isoformat() + str.format(
        'Z{0:+06.2f}', float(time.timezone) / 3600)
    endTime = end.isoformat() + str.format(
        'Z{0:+06.2f}', float(time.timezone) / 3600)
    history = conn.get_spot_price_history(
        start_time=startTime,
        end_time=endTime,
        instance_type=type,
        product_description='Linux/UNIX')

    min_price = history[0].price
    max_price = history[0].price
    for h in history:
        if h.price > max_price:
            max_price = h.price
        if h.price < min_price:
            min_price = h.price

    return min_price, max_price


def price_list(conn, ecus):
    ilist = []
    for itype in possible_instance_types:
        normp = get_quote(itype[1])
        if normp < 0:
            continue

        minp, maxp = get_spot_quote(conn, itype[1])
        num = int(round(float(ecus) / itype[0]))
        if num == 0:
            continue

        amin = 1000.0 * min(minp, normp) / itype[0]
        if amin >= 10.0:
            continue

        ilist.append([
            num, itype[1],
            num * itype[0],
            normp * num,
            minp * num,
            minp * num, amin])

    return sorted(ilist, key=lambda x: x[-1])


def print_price(ilist):
    for idx, i in enumerate(ilist):
        print '%03d: %3d x %-11s\t(%3d ECU)\t: normal $%3.2f/h \tspot $%.3f - $%5.2f\t min $%.3f /kECUh' % tuple([idx] + i)


def get_prices(args):
    ilist = sorted(price_list(connect(), args.ecus), key=lambda x: x[-1])
    print_price(ilist)


def ssh_call(inst, cmd, ret_out=True):
    pr = subprocess.Popen(
        ['ssh', '-i', '%s/.ssh/aws/kp_%s.pem' % (HOME, ZONE),
         '-o', 'StrictHostKeyChecking=no',
         '-o', 'LogLevel=ERROR',
         '-o', 'UserKnownHostsFile=/dev/null',
         'ubuntu@' + inst.ip_address, cmd],
        stdout=subprocess.PIPE if ret_out else None)

    output = ''
    while ret_out:
        line = pr.stdout.readline()
        if line == '':
            break
        output = output + line

    pr.wait()
    return output


def login_instance(args):
    conn = connect()
    instances = get_instances(conn, active_only=True)
    print 'Log into instance %s' % instances[args.id].id
    ssh_call(instances[args.id], '', False)


def terminate(args):
    conn = connect()
    instances = get_instances(conn, active_only=False)
    for i in instances:
        if i.state != "terminated":
            print "Terminating instance %s" % i.id
            i.terminate()
    spotInstances = conn.get_all_spot_instance_requests()
    for i in spotInstances:
        if i.state == 'active' or i.state == 'open':
            conn.cancel_spot_instance_requests([i.id])


def wait_for(instances):
    for i in instances:
        while i.state != u'running':
            time.sleep(1)
            i.update()


def create_instance(args):
    conn = connect()

    ilist = price_list(conn, args.ecus)
    print_price(ilist, False)

    cont = raw_input('(n)ormal, (s)pot, (a)bort ? ')
    if cont != 'n' and cont != 's':
        sys.exit(1)
    type_idx = int(raw_input('instance type number ? '))

    instance_type = ilist[type_idx][2]
    num_instances = ilist[type_idx][1]
    print('Choosing %dx \'%s\'' % (num_instances, instance_type))

    instances = []
    if cont == 'n':
        res = conn.run_instances(
            image_id=IMAGE_ID,
            min_count=num_instances,
            max_count=num_instances,
            key_name='kp_' + ZONE,
            security_groups=['sg_' + ZONE],
            instance_initiated_shutdown_behavior='stop',
            instance_type=instance_type)
        instances = res.instances

    elif cont == 's':
        price = float(raw_input('max price $')) / num_instances
        res = conn.request_spot_instances(
            key_name='kp_' + ZONE,
            security_groups=['sg_' + ZONE],
            image_id=IMAGE_ID,
            count=num_instances,
            price=str(price),
            instance_type=instance_type,
            type='one-time')
        print 'ok, check web for status'
        return

    else:
        sys.exit(1)

    print ('Waiting for instances...')
    wait_for(instances)
    list_status(instances)


def abort_scripts(args):
    conn = connect()
    instances = get_instances(conn, active_only=True)
    for inst in instances:
        print "Aborting scripts on instance '%s'..." % inst.id
        ssh_call(inst, 'killall python', False)


def update_git(args):
    conn = connect()
    instances = get_instances(conn, active_only=True)
    for inst in instances:
        print "Updating git repo on instance '%s'..." % inst.id
        cmd = "cd project/optimal-mental-rotation && git pull"
        ssh_call(inst, cmd, False)


def fetch_data(args):
    conn = connect()
    instances = get_instances(conn, active_only=True)
    print 'Fetch %s (%s) from instance %s' % (
        args.model, args.version, instances[args.id].id)

    addr = "ubuntu@%s" % instances[args.id].ip_address

    dpkg = path("data/model/%s_%s.dpkg" % (args.model, args.version))
    if dpkg.exists():
        cmd = ["git", "annex", "unlock", dpkg]
        subprocess.call(cmd)
        path(dpkg).rmtree_p()

    raw = path("data/sim-raw/%s/%s.tar.gz" % (args.model, args.version))
    if raw.islink():
        cmd = ["git", "annex", "unlock", raw]
        subprocess.call(cmd)
    if raw.exists():
        path(raw).remove()

    cmd = [
        "scp", "-r",
        "-i", "%s/.ssh/aws/kp_%s.pem" % (HOME, ZONE),
        "%s:project/optimal-mental-rotation/data/model/%s_%s.dpkg" % (
            addr, args.model, args.version),
        "data/model/"]
    subprocess.call(cmd)

    cmd = [
        "scp", "-r",
        "-i", "%s/.ssh/aws/kp_%s.pem" % (HOME, ZONE),
        "%s:project/optimal-mental-rotation/data/sim-raw/%s/%s.tar.gz" % (
            addr, args.model, args.version),
        "data/sim-raw/%s/" % args.model]
    subprocess.call(cmd)


if __name__ == "__main__":

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help='sub-command help')

    # list
    list_parser = subparsers.add_parser(
        'list', help="list instances")
    list_parser.set_defaults(func=list_instances)

    # price
    price_parser = subparsers.add_parser(
        'price', help="get price quotes")
    price_parser.add_argument(
        'ecus', type=int, help="number of ECUs")
    price_parser.set_defaults(func=get_prices)

    # create
    create_parser = subparsers.add_parser(
        'create', help="create ec2 instances")
    create_parser.add_argument(
        'ecus', type=int, help="number of desired ECUs")
    create_parser.set_defaults(func=create_instance)

    # # start
    # start_parser = subparsers.add_parser(
    #     'start', help="start simulations")
    # start_parser.add_argument(
    #     'id', type=int, required=True,
    #     help="which instance to start server on")
    # start_parser.set_defaults(func=start_scripts)

    # abort
    abort_parser = subparsers.add_parser(
        'abort', help="abort scripts on all nodes")
    abort_parser.set_defaults(func=abort_scripts)

    # update
    update_parser = subparsers.add_parser(
        'update', help="update git repositories on all nodes")
    update_parser.set_defaults(func=update_git)

    # fetch
    fetch_parser = subparsers.add_parser(
        'fetch', help="fetch git repositories on all nodes")
    fetch_parser.add_argument(
        'id', type=int,
        help="which instance to fetch data from")
    fetch_parser.add_argument(
        'model',
        help="which model data to get")
    fetch_parser.add_argument(
        'version',
        help="which data version to get")
    fetch_parser.set_defaults(func=fetch_data)

    # login
    login_parser = subparsers.add_parser(
        'login', help="log in to instance")
    login_parser.add_argument(
        'id', type=int,
        help="which instance to log in to")
    login_parser.set_defaults(func=login_instance)

    # terminate
    terminate_parser = subparsers.add_parser(
        'terminate', help="terminate all nodes")
    terminate_parser.set_defaults(func=terminate)

    args = parser.parse_args()
    args.func(args)
