# SPDX-License-Identifier: Apache-2.0
"""Keep a borrowed exclusive flock until an operator ends this process.

No device observation or quiescence verification is provided.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import stat


def verify(fd,path):
    actual=os.fstat(fd);expected=Path(path).lstat()
    if not stat.S_ISREG(expected.st_mode) or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise ValueError('canonical lock inode')
    text=Path('/proc/self/fdinfo',str(fd)).read_text()
    rows=re.findall(r'^lock:\s*\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+\d+\s+[0-9a-fA-F]+:[0-9a-fA-F]+:(\d+)\s+0\s+EOF\s*$',text,re.MULTILINE)
    if str(actual.st_ino) not in rows:raise ValueError('borrowed exclusive lock missing')


def main():
    p=argparse.ArgumentParser();p.add_argument('--fd',type=int,required=True);p.add_argument('--path',required=True);p.add_argument('--ready-fd',type=int,required=True);args=p.parse_args()
    verify(args.fd,args.path)
    if os.write(args.ready_fd,b'Q')!=1:raise RuntimeError('ready receipt')
    os.close(args.ready_fd)
    while True:signal.pause()
if __name__=='__main__':main()
