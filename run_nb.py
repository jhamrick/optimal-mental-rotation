#!/usr/bin/env python
"""
Simple example script for running notebooks and checking for errors

See https://gist.github.com/minrk/2620876

Usage: `run_nb.py foo.ipynb [bar.ipynb [...]]`

"""

import os
import sys
from IPython.zmq.blockingkernelmanager import BlockingKernelManager
from IPython.nbformat.current import reads


def run_notebook(nb):
    km = BlockingKernelManager()
    km.start_kernel(stderr=open(os.devnull, 'w'))
    km.start_channels()
    shell = km.shell_channel
    # simple ping:
    shell.execute("pass")
    shell.get_msg()

    cells = 0
    failures = 0
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            shell.execute(cell.input)
            # wait for finish, maximum 20s
            reply = shell.get_msg(timeout=20)['content']
            if reply['status'] == 'error':
                failures += 1
                print "\nFAILURE:"
                print cell.input
                print '-----'
                print "raised:"
                print '\n'.join(reply['traceback'])
            cells += 1
            sys.stdout.write('.')

    print
    print "ran notebook %s" % nb.metadata.name
    print "    ran %3i cells" % cells
    if failures:
        print "    %3i cells raised exceptions" % failures
    km.shutdown_kernel()
    del km

if __name__ == '__main__':
    for ipynb in sys.argv[1:]:
        print "running %s" % ipynb
        with open(ipynb) as f:
            nb = reads(f.read(), 'json')
        run_notebook(nb)
