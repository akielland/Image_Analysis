(base)
anderss - MacBook - Pro: ~ anders$ scp / Users / anders / Documents / IN4310 / mandatory / scr01 / main06.py
kielland @ login.ifi.uio.no: // uio / hume / student - u45 / kielland / Desktop
kielland @ login.ifi.uio.no
's password:
main06.py
100 % 28
KB
634.2
KB / s
00: 00
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
login: Thu
Mar
16
21: 32:55
2023
from ti0022q162

-7197.
bb.online.no
Velkommen
til
afram.ifi.uio.no.
afram
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den
egner
seg
ikke
til
simuleringer
og
tunge
jobber.
Vær
sparsom
med
ressurser
og
vis
hensyn.
Ved
spørsmål: Kontakt
drift @ ifi.uio.no.
---
Welcome
to
afram.ifi.uio.no.
afram is a
shared
machine
for all IFI users.
You
should
not run
simulations and heavy
jobs
on
this
machine.
Use
resources
sparingly and show
consideration
for other users.
    If
    you
    have
    questions, send
    email
    to
    drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ afram ~]$ ls
ANDERS
Documents
Mail
Pictures
'Shortcut to Removable Disk (F).lnk'
WINDOWS
mail
reglog.txt
AdobeWeb.log
Downloads
Music
Public
Templates
dead.letter
mbox
update
Desktop
'Folder Settings'  'Network Trash Folder'
SCALE - 920228.
epc
Videos
desktop.ini
pc
'~'
[kielland @ afram ~]$ pwd
/ uio / hume / student - u45 / kielland
[kielland @ afram
 ~]$ scp / uio / hume / student - u45 / kielland / Desktop / main06.pykielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
usage: scp[-346
BCpqrTv] [-c cipher][-F
ssh_config] [-i identity_file]
[-J destination][-l
limit] [-o ssh_option][-P
port]
[-S program]
source...target
[kielland @ afram ~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
# Current resource situation on the ML nodes, updated 2023-03-17 08:00:01
#             memory(GiB)    load                  /scratch(GiB)   GPU
# name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
ml1.hpc.uio.no
125
121
0.00
0.03
0.00
1023
49 % 0 % | 0 %
ml2.hpc.uio.no
125
102
32.63
32.80
33.37
1023
53 % 19 % | 12.75 %
ml3.hpc.uio.no
125
85
21.96
22.52
22.40
2047
53 % 16.75 % | 11 %
ml6.hpc.uio.no
251
123
20.21
20.57
21.34
5235
18 % 14.5 % | 4.875 %
ml7.hpc.uio.no
251
174
38.49
38.34
38.41
5213
21 % 7.25 % | 0.375 %
ml8.hpc.uio.no
1007
884
100.64
159.20
156.16
5118
43 % 81.5 % | 33.75 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From
22
nd
Feb
to
March
19
th
and
April
7
th
to
April
29
th

Any
none
course
related
analysis
during
the
course
hours
will
be
terminated
without
notice

This
machine
was
rebooted
13: 24, 26
Feb
2023

___
Last
login: Fri
Mar
17
07: 53:20
2023
from vdi

-dhcp - 96 - 131.
uio.no
[kielland @ ml6 ~]$ ls
main06.py
[kielland @ ml6 ~]$ python - -version
python: Command
not found.
[kielland @ ml6 ~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
[kielland @ ml6 ~]$ python - -version
Python
3.9
.7
[kielland @ ml6 ~]$ main06.py
main06.py: Command
not found.
[kielland @ ml6 ~]$ python
main06.py
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main06.py", line
13, in < module >
import efficientnet_pytorch as efn

ModuleNotFoundError: No
module
named
'efficientnet_pytorch'
[kielland @ ml6 ~]$ exit
logout
Connection
to
ml6.hpc.uio.no
closed.
[kielland @ afram ~]$ exit
logout
Connection
to
login.ifi.uio.no
closed.
(base)
anderss - MacBook - Pro: ~ anders$ scp / Users / anders / Documents / IN4310 / mandatory / scr01 / main06.py
kielland @ login.ifi.uio.no: // uio / hume / student - u45 / kielland / Desktop
kielland @ login.ifi.uio.no
's password:
main06.py
100 % 28
KB
584.3
KB / s
00: 00
(base)
anderss - MacBook - Pro: ~ anders$ anders$ ssh
kielland @ login.ifi.uio.no
-bash: anders$: command
not found
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
login: Fri
Mar
17
0
8: 24:0
8
2023
from ti0022q162

-7197.
bb.online.no
Velkommen
til
afram.ifi.uio.no.
afram
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den
egner
seg
ikke
til
simuleringer
og
tunge
jobber.
Vær
sparsom
med
ressurser
og
vis
hensyn.
Ved
spørsmål: Kontakt
drift @ ifi.uio.no.
---
Welcome
to
afram.ifi.uio.no.
afram is a
shared
machine
for all IFI users.
You
should
not run
simulations and heavy
jobs
on
this
machine.
Use
resources
sparingly and show
consideration
for other users.
    If
    you
    have
    questions, send
    email
    to
    drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ afram
 ~]$ scp / uio / hume / student - u45 / kielland / Desktop / main06.pykielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
usage: scp[-346
BCpqrTv] [-c cipher][-F
ssh_config] [-i identity_file]
[-J destination][-l
limit] [-o ssh_option][-P
port]
[-S program]
source...target
[kielland @ afram ~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
# Current resource situation on the ML nodes, updated 2023-03-17 08:30:02
#             memory(GiB)    load                  /scratch(GiB)   GPU
# name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
ml1.hpc.uio.no
125
121
0.08
0.06
0.01
1023
49 % 0 % | 0 %
ml2.hpc.uio.no
125
82
34.74
38.68
38.12
1023
53 % 42.75 % | 17.75 %
ml3.hpc.uio.no
125
87
22.61
22.33
22.39
2047
53 % 46 % | 26 %
ml6.hpc.uio.no
251
139
40.79
43.65
35.47
5235
18 % 24.875 % | 11.625 %
ml7.hpc.uio.no
251
174
36.65
38.10
38.39
5213
21 % 7.5 % | 0.375 %
ml8.hpc.uio.no
1007
883
124.54
115.72
122.64
5118
43 % 80.75 % | 63 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From
22
nd
Feb
to
March
19
th
and
April
7
th
to
April
29
th

Any
none
course
related
analysis
during
the
course
hours
will
be
terminated
without
notice

This
machine
was
rebooted
13: 24, 26
Feb
2023

___
Last
login: Fri
Mar
17
0
8: 29:20
2023
from afram.ifi.uio.no

[kielland @ ml6 ~]$ ls
main06.py
[kielland @ ml6 ~]$ kielland @ ml6
~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
Unknown
user:]$.
[kielland @ ml6 ~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
[kielland @ ml6
~]$ python - -version
Python
3.9
.7
[kielland @ ml6
~]$ python
main06.py
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main06.py", line
13, in < module >
import efficientnet_pytorch as efn

ModuleNotFoundError: No
module
named
'efficientnet_pytorch'
[kielland @ ml6
~]$ pip
install
efficientnet_pytorch
Defaulting
to
user
installation
because
normal
site - packages is not writeable
Collecting
efficientnet_pytorch
Downloading
efficientnet_pytorch - 0.7
.1.tar.gz(21
kB)
Preparing
metadata(setup.py)...done
Requirement
already
satisfied: torch in / storage / software / PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised / lib / python3
.9 / site - packages(
from efficientnet_pytorch) (1.10.0)
Requirement
already
satisfied: typing - extensions in / storage / software / PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised / lib / python3
.9 / site - packages(
from torch->efficientnet_pytorch) (4.0.0)
Building
wheels
for collected packages: efficientnet - pytorch
Building
wheel
for efficientnet - pytorch(setup.py)...done
Created wheel for efficientnet-pytorch: filename = efficientnet_pytorch - 0.7
.1 - py3 - none - any.whl
size = 16446
sha256 = 3970
b0f39d7c4a3975a8ee60ea83737bc095a7aef7acfec67179b55a4f8bdc3f
Stored in directory: / itf - fi - ml / home / kielland /.cache / pip / wheels / 29 / 16 / 24 / 752e89
d88d333af39a288421e64d613b5f652918e39ef1f8e3
Successfully
built
efficientnet - pytorch
Installing
collected
packages: efficientnet - pytorch
Successfully
installed
efficientnet - pytorch - 0.7
.1
[kielland @ ml6
~]$ python
main06.py
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main06.py", line
557, in < module >
          train_files, val_files, test_files, train_labels, val_labels, test_labels = split_data(
    reduced_num_samples=True)
File
"/itf-fi-ml/home/kielland/main06.py", line
23, in split_data
subdirs = [x for x in os.listdir(data_dir) if not x.startswith('.')]  # skip hidden files/directories
FileNotFoundError: [Errno 2]
No
such
file or directory: '/Users/anders/Documents/IN4310/mandatory/mandatory1_data/'
[kielland @ ml6
~]$ ls
main06.py
[kielland @ ml6
~]$ exit
logout
Connection
to
ml6.hpc.uio.no
closed.
[kielland @ afram
~]$ exit
logout
Connection
to
login.ifi.uio.no
closed.
(base)
anderss - MacBook - Pro: ~ anders$ scp / Users / anders / Documents / IN4310 / mandatory / scr01 / main07.py
kielland @ login.ifi.uio.no: // uio / hume / student - u45 / kielland / Desktop
kielland @ login.ifi.uio.no
's password:
main07.py
100 % 28
KB
538.7
KB / s
00: 00
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
login: Fri
Mar
17
0
8: 33:34
2023
from ti0022q162

-7197.
bb.online.no
Velkommen
til
afram.ifi.uio.no.
afram
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den egner seg ikke til simuleringer og tunge jobber.
Vær sparsom med ressurser og vis hensyn.
Ved spørsmål: Kontakt
drift @ ifi.uio.no.
- --
Welcome
to
afram.ifi.uio.no.
afram is a
shared
machine
for all IFI users.
You should not run simulations and heavy jobs on this machine.
Use resources sparingly and show consideration for other users.
If you have questions, send email to drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ afram
~]$ scp / uio / hume / student - u45 / kielland / Desktop / main07.pykielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
usage: scp[-346
BCpqrTv] [-c cipher][-F
ssh_config] [-i identity_file]
[-J
destination] [-l limit][-o
ssh_option] [-P port]
[-S
program] source...target
[kielland @ afram
~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
          # Current resource situation on the ML nodes, updated 2023-03-17 08:30:02
          #             memory(GiB)    load                  /scratch(GiB)   GPU
          # name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
          ml1.hpc.uio.no
125
121
0.08
0.06
0.01
1023
49 % 0 % | 0 %
ml2.hpc.uio.no
125
82
34.74
38.68
38.12
1023
53 % 42.75 % | 17.75 %
ml3.hpc.uio.no
125
87
22.61
22.33
22.39
2047
53 % 46 % | 26 %
ml6.hpc.uio.no
251
139
40.79
43.65
35.47
5235
18 % 24.875 % | 11.625 %
ml7.hpc.uio.no
251
174
36.65
38.10
38.39
5213
21 % 7.5 % | 0.375 %
ml8.hpc.uio.no
1007
883
124.54
115.72
122.64
5118
43 % 80.75 % | 63 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From 22nd Feb to March 19th
and
April 7th to April 29th

Any none course related analysis during the course hours
will be terminated without notice


This machine was rebooted 13:24, 26
Feb
2023

___
Last
login: Fri
Mar
17
0
8: 34:20
2023
from afram.ifi.uio.no

[kielland @ ml6 ~]$ ls
main06.py
[kielland @ ml6
~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
[kielland @ ml6
~]$ python
main07.py
python: can
't open file ' / itf - fi - ml / home / kielland / main07.py
': [Errno 2] No such file or directory
[kielland @ ml6
~]$ ls
main06.py
[kielland @ ml6
~]$ exit
logout
Connection
to
ml6.hpc.uio.no
closed.
[kielland @ afram
~]$ scp / uio / hume / student - u45 / kielland / Desktop / main07.pykielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
usage: scp[-346
BCpqrTv] [-c cipher][-F
ssh_config] [-i identity_file]
[-J
destination] [-l limit][-o
ssh_option] [-P port]
[-S
program] source...target
[kielland @ afram
~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
          # Current resource situation on the ML nodes, updated 2023-03-17 08:30:02
          #             memory(GiB)    load                  /scratch(GiB)   GPU
          # name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
          ml1.hpc.uio.no
125
121
0.08
0.06
0.01
1023
49 % 0 % | 0 %
ml2.hpc.uio.no
125
82
34.74
38.68
38.12
1023
53 % 42.75 % | 17.75 %
ml3.hpc.uio.no
125
87
22.61
22.33
22.39
2047
53 % 46 % | 26 %
ml6.hpc.uio.no
251
139
40.79
43.65
35.47
5235
18 % 24.875 % | 11.625 %
ml7.hpc.uio.no
251
174
36.65
38.10
38.39
5213
21 % 7.5 % | 0.375 %
ml8.hpc.uio.no
1007
883
124.54
115.72
122.64
5118
43 % 80.75 % | 63 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From 22nd Feb to March 19th
and
April 7th to April 29th

Any none course related analysis during the course hours
will be terminated without notice


This machine was rebooted 13:24, 26
Feb
2023

___
Last
login: Fri
Mar
17
0
8: 51:34
2023
from afram.ifi.uio.no

[kielland @ ml6 ~]$ ls
main06.py
[kielland @ ml6
~]$ rm
main06.py
[kielland @ ml6
~]$ ls
[kielland @ ml6
~]$ exit
logout
Connection
to
ml6.hpc.uio.no
closed.
[kielland @ afram
~]$ scp / uio / hume / student - u45 / kielland / Desktop / main07.pykielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
usage: scp[-346
BCpqrTv] [-c cipher][-F
ssh_config] [-i identity_file]
[-J
destination] [-l limit][-o
ssh_option] [-P port]
[-S
program] source...target
[kielland @ afram
~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
          # Current resource situation on the ML nodes, updated 2023-03-17 08:30:02
          #             memory(GiB)    load                  /scratch(GiB)   GPU
          # name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
          ml1.hpc.uio.no
125
121
0.08
0.06
0.01
1023
49 % 0 % | 0 %
ml2.hpc.uio.no
125
82
34.74
38.68
38.12
1023
53 % 42.75 % | 17.75 %
ml3.hpc.uio.no
125
87
22.61
22.33
22.39
2047
53 % 46 % | 26 %
ml6.hpc.uio.no
251
139
40.79
43.65
35.47
5235
18 % 24.875 % | 11.625 %
ml7.hpc.uio.no
251
174
36.65
38.10
38.39
5213
21 % 7.5 % | 0.375 %
ml8.hpc.uio.no
1007
883
124.54
115.72
122.64
5118
43 % 80.75 % | 63 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From 22nd Feb to March 19th
and
April 7th to April 29th

Any none course related analysis during the course hours
will be terminated without notice


This machine was rebooted 13:24, 26
Feb
2023

___
Last
login: Fri
Mar
17
0
8: 53:03
2023
from afram.ifi.uio.no

[kielland @ ml6 ~]$ ls
[kielland @ ml6
~]$ exit
logout
Connection
to
ml6.hpc.uio.no
closed.
[kielland @ afram
~]$ exit
logout
Connection
to
login.ifi.uio.no
closed.
(base)
anderss - MacBook - Pro: ~ anders$ scp / Users / anders / Documents / IN4310 / mandatory / scr01 / main07.py
kielland @ login.ifi.uio.no: // uio / hume / student - u45 / kielland / Desktop
kielland @ login.ifi.uio.no
's password:
main07.py
100 % 28
KB
723.4
KB / s
00: 00
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
login: Mon
Mar
6
11: 51:30
2023
from ti0022q162

-2362.
bb.online.no
Velkommen
til
aftur.ifi.uio.no.
aftur
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den egner seg ikke til simuleringer og tunge jobber.
Vær sparsom med ressurser og vis hensyn.
Ved spørsmål: Kontakt
drift @ ifi.uio.no.
- --
Welcome
to
aftur.ifi.uio.no.
aftur is a
shared
machine
for all IFI users.
You should not run simulations and heavy jobs on this machine.
Use resources sparingly and show consideration for other users.
If you have questions, send email to drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ aftur
~]$ scp / uio / hume / student - u45 / kielland / Desktop / main07.py
kielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
kielland @ ml6.hpc.uio.no
's password:
main07.py
100 % 28
KB
8.2
MB / s
00: 00
[kielland @ aftur
~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
          # Current resource situation on the ML nodes, updated 2023-03-17 08:30:02
          #             memory(GiB)    load                  /scratch(GiB)   GPU
          # name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
          ml1.hpc.uio.no
125
121
0.08
0.06
0.01
1023
49 % 0 % | 0 %
ml2.hpc.uio.no
125
82
34.74
38.68
38.12
1023
53 % 42.75 % | 17.75 %
ml3.hpc.uio.no
125
87
22.61
22.33
22.39
2047
53 % 46 % | 26 %
ml6.hpc.uio.no
251
139
40.79
43.65
35.47
5235
18 % 24.875 % | 11.625 %
ml7.hpc.uio.no
251
174
36.65
38.10
38.39
5213
21 % 7.5 % | 0.375 %
ml8.hpc.uio.no
1007
883
124.54
115.72
122.64
5118
43 % 80.75 % | 63 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From 22nd Feb to March 19th
and
April 7th to April 29th

Any none course related analysis during the course hours
will be terminated without notice


This machine was rebooted 13:24, 26
Feb
2023

___
Last
login: Fri
Mar
17
0
8: 54:41
2023
from afram.ifi.uio.no

[kielland @ ml6 ~]$ ls
main07.py
[kielland @ ml6
~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
[kielland @ ml6
~]$ python
main07.py
The
number
of
samples in each
data
set
Train
set: 496
samples
Validation
set: 200
samples
Test
set: 300
samples
- ---------
Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
to / itf - fi - ml / home / kielland /.cache / torch / hub / checkpoints / efficientnet - b0 - 355
c32eb.pth
100 % |██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | 20.4
M / 20.4
M[00:00 < 00:00, 44.6
MB / s]
Loaded
pretrained
weights
for efficientnet - b0
Epoch 1 / 2
current mean of losses  1.8182744979858398
exit
^ C
^ CTraceback (most recent call last):
    File
"/itf-fi-ml/home/kielland/main07.py", line
574, in < module >
          losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val = run_training()
File
"/itf-fi-ml/home/kielland/main07.py", line
199, in run_training
train_model(train_loader, val_loader, model, criterion, optimizer, device)
File
"/itf-fi-ml/home/kielland/main07.py", line
136, in train_model
losses = train_epoch(model, train_loader, criterion, device, optimizer)
File
"/itf-fi-ml/home/kielland/main07.py", line
106, in train_epoch
output = model(inputs)  # output tensor = prediction scores (logits) for each class in the output space
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/itf-fi-ml/home/kielland/.local/lib/python3.9/site-packages/efficientnet_pytorch/model.py", line
314, in forward
x = self.extract_features(inputs)
File
"/itf-fi-ml/home/kielland/.local/lib/python3.9/site-packages/efficientnet_pytorch/model.py", line
296, in extract_features
x = block(x, drop_connect_rate=drop_connect_rate)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/itf-fi-ml/home/kielland/.local/lib/python3.9/site-packages/efficientnet_pytorch/model.py", line
109, in forward
x = self._depthwise_conv(x)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/itf-fi-ml/home/kielland/.local/lib/python3.9/site-packages/efficientnet_pytorch/utils.py", line
275, in forward
x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
KeyboardInterrupt

[kielland @ ml6 ~]$ exit
logout
Connection
to
ml6.hpc.uio.no
closed.
[kielland @ aftur ~]$ exit
logout
Connection
to
login.ifi.uio.no
closed.
(base)
anderss - MacBook - Pro: ~ anders$ scp / Users / anders / Documents / IN4310 / mandatory / scr01 / main07.py
kielland @ login.ifi.uio.no: // uio / hume / student - u45 / kielland / Desktop
kielland @ login.ifi.uio.no
's password:
main07.py
100 % 28
KB
761.1
KB / s
00: 00
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
login: Fri
Mar
17
0
8: 55:28
2023
from ti0022q162

-7197.
bb.online.no
Velkommen
til
aftur.ifi.uio.no.
aftur
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den
egner
seg
ikke
til
simuleringer
og
tunge
jobber.
Vær
sparsom
med
ressurser
og
vis
hensyn.
Ved
spørsmål: Kontakt
drift @ ifi.uio.no.
---
Welcome
to
aftur.ifi.uio.no.
aftur is a
shared
machine
for all IFI users.
You
should
not run
simulations and heavy
jobs
on
this
machine.
Use
resources
sparingly and show
consideration
for other users.
    If
    you
    have
    questions, send
    email
    to
    drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ aftur ~]$ ls
ANDERS
Documents
Mail
Pictures
'Shortcut to Removable Disk (F).lnk'
WINDOWS
mail
reglog.txt
AdobeWeb.log
Downloads
Music
Public
Templates
dead.letter
mbox
update
Desktop
'Folder Settings'  'Network Trash Folder'
SCALE - 920228.
epc
Videos
desktop.ini
pc
'~'
[kielland @ aftur ~]$ scp / uio / hume / student - u45 / kielland / Desktop / main07.py
kielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
kielland @ ml6.hpc.uio.no
's password:
main07.py
100 % 28
KB
7.8
MB / s
00: 00
[kielland @ aftur ~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
# Current resource situation on the ML nodes, updated 2023-03-17 09:00:01
#             memory(GiB)    load                  /scratch(GiB)   GPU
# name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
ml1.hpc.uio.no
125
104
12.05
12.11
10.37
1023
49 % 10.5 % | 5.25 %
ml2.hpc.uio.no
125
80
35.28
36.26
38.01
1023
53 % 47.25 % | 18.25 %
ml3.hpc.uio.no
125
65
34.06
33.87
30.37
2047
53 % 45.75 % | 27.75 %
ml6.hpc.uio.no
251
111
88.05
65.01
48.94
5235
18 % 34.25 % | 16.25 %
ml7.hpc.uio.no
251
156
50.58
50.57
48.69
5213
21 % 23.5 % | 11.25 %
ml8.hpc.uio.no
1007
883
68.78
92.85
107.67
5118
43 % 82.75 % | 58.75 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From
22
nd
Feb
to
March
19
th
and
April
7
th
to
April
29
th

Any
none
course
related
analysis
during
the
course
hours
will
be
terminated
without
notice

This
machine
was
rebooted
13: 24, 26
Feb
2023

___
Last
login: Fri
Mar
17
0
8: 56:25
2023
from aftur.ifi.uio.no

[kielland @ ml6 ~]$ ls
main07.py
[kielland @ ml6 ~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
[kielland @ ml6 ~]$ CUDA_VISIBLE_DEVICES = 4
python
main07.py
CUDA_VISIBLE_DEVICES = 4: Command
not found.
[kielland @ ml6 ~]$ export
CUDA_VISIBLE_DEVICES = 4
python
main07.py
export: Command
not found.
[kielland @ ml6 ~]$ which
CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES: Command
not found.
[kielland @ ml6 ~]$ nvidia - smi
Fri
Mar
17
0
9: 15:19
2023
+-----------------------------------------------------------------------------+
| NVIDIA - SMI
515.86
.01
Driver
Version: 515.86
.01
CUDA
Version: 11.7 |
| -------------------------------+----------------------+----------------------+
| GPU
Name
Persistence - M | Bus - Id
Disp.A | Volatile
Uncorr.ECC |
| Fan
Temp
Perf
Pwr: Usage / Cap | Memory - Usage | GPU - Util
Compute
M. |
| | | MIG
M. |
|= == == == == == == == == == == == == == == == += == == == == == == == == == == = += == == == == == == == == == == = |
| 0
NVIDIA
GeForce...Off | 00000000: 01:00.0
Off | N / A |
| 25 % 46
C
P2
63
W / 250
W | 7038
MiB / 11264
MiB | 10 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 1
NVIDIA
GeForce...Off | 00000000: 23:00.0
Off | N / A |
| 27 % 47
C
P2
72
W / 250
W | 4407
MiB / 11264
MiB | 5 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 2
NVIDIA
GeForce...Off | 00000000: 41:00.0
Off | N / A |
| 25 % 45
C
P2
60
W / 250
W | 4483
MiB / 11264
MiB | 5 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 3
NVIDIA
GeForce...Off | 00000000: 61:00.0
Off | N / A |
| 22 % 40
C
P2
47
W / 250
W | 4483
MiB / 11264
MiB | 5 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 4
NVIDIA
GeForce...Off | 00000000: 81:00.0
Off | N / A |
| 25 % 44
C
P2
73
W / 250
W | 7442
MiB / 11264
MiB | 5 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 5
NVIDIA
GeForce...Off | 00000000: A1:00.0
Off | N / A |
| 22 % 31
C
P8
21
W / 250
W | 3
MiB / 11264
MiB | 0 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 6
NVIDIA
GeForce...Off | 00000000: C1:00.0
Off | N / A |
| 22 % 31
C
P8
28
W / 250
W | 5
MiB / 11264
MiB | 0 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 7
NVIDIA
GeForce...Off | 00000000: E1:00.0
Off | N / A |
| 22 % 28
C
P8
25
W / 250
W | 5
MiB / 11264
MiB | 0 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes: |
| GPU
GI
CI
PID
Type
Process
name
GPU
Memory |
| ID
ID
Usage |
|= == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == |
| 0
N / A
N / A
345659
C
python3
2555
MiB |
| 0
N / A
N / A
1732903
C
python3
3561
MiB |
| 0
N / A
N / A
2349494
C
python
919
MiB |
| 1
N / A
N / A
1733204
C...a / envs / lfdeep / bin / python3
3485
MiB |
| 1
N / A
N / A
2349711
C... / envs / mariaenv / bin / python
919
MiB |
| 2
N / A
N / A
1733216
C...a / envs / lfdeep / bin / python3
3561
MiB |
| 2
N / A
N / A
2349712
C... / envs / mariaenv / bin / python
919
MiB |
| 3
N / A
N / A
1733260
C...a / envs / lfdeep / bin / python3
3561
MiB |
| 3
N / A
N / A
2349713
C... / envs / mariaenv / bin / python
919
MiB |
| 4
N / A
N / A
54561
C
python3
7439
MiB |
+-----------------------------------------------------------------------------+
[kielland @ ml6 ~]$ export
CUDA_VISIBLE_DEVICES = 5
export: Command
not found.
[kielland @ ml6 ~]$ CUDA_VISIBLE_DEVICES = 5
python
main.py
CUDA_VISIBLE_DEVICES = 5: Command
not found.
[kielland @ ml6 ~]$ CUDA_VISIBLE_DEVICES = 5
python
main07.py
CUDA_VISIBLE_DEVICES = 5: Command
not found.
[kielland @ ml6 ~]$ $ sudo
export
CUDA_VISIBLE_DEVICES = 5
$: Command
not found.
[kielland @ ml6 ~]$ python - -version
Python
3.9
.7
[kielland @ ml6 ~]$ htop
[kielland @ ml6 ~]$ exit
logout
Connection
to
ml6.hpc.uio.no
closed.
[kielland @ aftur ~]$ exit
logout
Connection
to
login.ifi.uio.no
closed.
(base)
anderss - MacBook - Pro: ~ anders$ scp / Users / anders / Documents / IN4310 / mandatory / scr01 / main07.py
kielland @ login.ifi.uio.no: // uio / hume / student - u45 / kielland / Desktop
kielland @ login.ifi.uio.no
's password:
Permission
denied, please
try again.
kielland @ login.ifi.uio.no
's password:
main07.py
100 % 29
KB
701.7
KB / s
00: 00
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
failed
login: Fri
Mar
17
10: 22:24
CET
2023
from ti0022q162

-7197.
bb.online.no
on
ssh: notty
There
was
1
failed
login
attempt
since
the
last
successful
login.
Last
login: Fri
Mar
17
0
9: 04:50
2023
from ti0022q162

-7197.
bb.online.no
Velkommen
til
aftur.ifi.uio.no.
aftur
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den
egner
seg
ikke
til
simuleringer
og
tunge
jobber.
Vær
sparsom
med
ressurser
og
vis
hensyn.
Ved
spørsmål: Kontakt
drift @ ifi.uio.no.
---
Welcome
to
aftur.ifi.uio.no.
aftur is a
shared
machine
for all IFI users.
You
should
not run
simulations and heavy
jobs
on
this
machine.
Use
resources
sparingly and show
consideration
for other users.
    If
    you
    have
    questions, send
    email
    to
    drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ aftur ~]$ scp / uio / hume / student - u45 / kielland / Desktop / main07.py
kielland @ ml6.hpc.uio.no: / itf - fi - ml / home / kielland
kielland @ ml6.hpc.uio.no
's password:
main07.py
100 % 29
KB
9.0
MB / s
00: 00
[kielland @ aftur ~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Permission
denied, please
try again.
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
# Current resource situation on the ML nodes, updated 2023-03-17 10:00:01
#             memory(GiB)    load                  /scratch(GiB)   GPU
# name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
ml1.hpc.uio.no
125
104
12.28
12.19
12.11
1023
49 % 10.5 % | 5 %
ml2.hpc.uio.no
125
95
37.51
39.87
39.08
1023
53 % 45.25 % | 16.75 %
ml3.hpc.uio.no
125
64
33.91
33.74
33.75
2047
53 % 32.5 % | 14 %
ml6.hpc.uio.no
251
127
68.57
67.90
63.68
5235
18 % 13.25 % | 9.125 %
ml7.hpc.uio.no
251
155
49.75
49.87
49.97
5213
21 % 20 % | 8.125 %
ml8.hpc.uio.no
1007
903
50.48
44.70
55.32
5118
43 % 72.75 % | 42.5 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From
22
nd
Feb
to
March
19
th
and
April
7
th
to
April
29
th

Any
none
course
related
analysis
during
the
course
hours
will
be
terminated
without
notice

This
machine
was
rebooted
13: 24, 26
Feb
2023

___
Last
failed
login: Fri
Mar
17
10: 24:07
CET
2023
from aftur.ifi.uio.no on

ssh: notty
There
was
1
failed
login
attempt
since
the
last
successful
login.
Last
login: Fri
Mar
17
0
9: 05:26
2023
from aftur.ifi.uio.no

[kielland @ ml6 ~]$ ls
main07.py
[kielland @ ml6 ~]$ nano
main07.py
[kielland @ ml6 ~]$ python - -version
python: Command
not found.
[kielland @ ml6 ~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
[kielland @ ml6 ~]$ python - -version
Python
3.9
.7
[kielland @ ml6 ~]$ CUDA_VISIBLE_DEVICES = 5
python
main07.py
CUDA_VISIBLE_DEVICES = 5: Command
not found.
[kielland @ ml6 ~]$ export
CUDA_VISIBLE_DEVICES = 5
export: Command
not found.
[kielland @ ml6 ~]$ client_loop: send
disconnect: Broken
pipe
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
login: Fri
Mar
17
06: 51:59
2023
from ti0022q162

-7197.
bb.online.no
Velkommen
til
nidur.ifi.uio.no.
nidur
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den
egner
seg
ikke
til
simuleringer
og
tunge
jobber.
Vær
sparsom
med
ressurser
og
vis
hensyn.
Ved
spørsmål: Kontakt
drift @ ifi.uio.no.
---
Welcome
to
nidur.ifi.uio.no.
nidur is a
shared
machine
for all IFI users.
You
should
not run
simulations and heavy
jobs
on
this
machine.
Use
resources
sparingly and show
consideration
for other users.
    If
    you
    have
    questions, send
    email
    to
    drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ nidur ~]$ ssh
kielland @ ml7.hpc.uio.no
kielland @ ml7.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
# Current resource situation on the ML nodes, updated 2023-03-17 12:30:02
#             memory(GiB)    load                  /scratch(GiB)   GPU
# name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
ml1.hpc.uio.no
125
91
29.61
29.07
21.75
1023
49 % 23.25 % | 13 %
ml2.hpc.uio.no
125
93
26.74
26.84
26.41
1023
53 % 46.5 % | 17.75 %
ml3.hpc.uio.no
125
103
11.36
11.39
12.47
2047
53 % 7.75 % | 3.25 %
ml6.hpc.uio.no
251
123
20.80
23.36
24.39
5235
18 % 50 % | 0 %
ml7.hpc.uio.no
251
168
44.02
42.59
40.83
5213
21 % 16 % | 4.625 %
ml8.hpc.uio.no
1007
879
19.59
36.50
43.37
5118
43 % 83.25 % | 29.25 %
#
# We encourage you to choose the least busy machine.
Last
login: Fri
Mar
17
06: 39:30
2023
from nidur.ifi.uio.no

[kielland @ ml7 ~]$ source
~ /.bashrc
if: Expression
Syntax.
then: Command
not found.
[kielland @ ml7 ~]$ which
bash
/ usr / bin / bash
[kielland @ ml7 ~]$ client_loop: send
disconnect: Broken
pipe
(base)
anderss - MacBook - Pro: ~ anders$ ssh
kielland @ login.ifi.uio.no
kielland @ login.ifi.uio.no
's password:
Last
login: Fri
Mar
17
12: 44:57
2023
from eduroam

-193 - 157 - 164 - 139.
wlan.uio.no
Velkommen
til
nidur.ifi.uio.no.
nidur
er
en
maskin
som
skal
fungere
som
felles
ressurs
for alle på IFI.
Den
egner
seg
ikke
til
simuleringer
og
tunge
jobber.
Vær
sparsom
med
ressurser
og
vis
hensyn.
Ved
spørsmål: Kontakt
drift @ ifi.uio.no.
---
Welcome
to
nidur.ifi.uio.no.
nidur is a
shared
machine
for all IFI users.
You
should
not run
simulations and heavy
jobs
on
this
machine.
Use
resources
sparingly and show
consideration
for other users.
    If
    you
    have
    questions, send
    email
    to
    drift @ ifi.uio.no
manpath: can
't set the locale; make sure $LC_* and $LANG are correct
[kielland @ nidur ~]$ ssh
kielland @ ml6.hpc.uio.no
kielland @ ml6.hpc.uio.no
's password:
Documentation
https: // www.uio.no / tjenester / it / forskning / kompetansehuber / uio - ai - hub - node - project / it - resources / ml - nodes /
# Current resource situation on the ML nodes, updated 2023-03-17 13:00:01
#             memory(GiB)    load                  /scratch(GiB)   GPU
# name          total  free    1-min  5-min 15-min  total  %used    Load|Memory
ml1.hpc.uio.no
125
104
15.45
20.11
19.54
1023
49 % 14.75 % | 6 %
ml2.hpc.uio.no
125
92
26.89
26.68
26.67
1023
53 % 45 % | 18.25 %
ml3.hpc.uio.no
125
103
11.38
11.39
11.53
2047
53 % 22.5 % | 7.75 %
ml6.hpc.uio.no
251
122
20.45
23.62
24.52
5235
18 % 4 % | 0.875 %
ml7.hpc.uio.no
251
170
41.16
41.50
42.02
5213
21 % 15 % | 4.375 %
ml8.hpc.uio.no
1007
877
46.64
45.68
46.62
5118
43 % 81.5 % | 53 %
#
# We encourage you to choose the least busy machine.
___

This
node(ml6.hpc.uio) is reserved
for a course(IN5400)

From
22
nd
Feb
to
March
19
th
and
April
7
th
to
April
29
th

Any
none
course
related
analysis
during
the
course
hours
will
be
terminated
without
notice

This
machine
was
rebooted
13: 24, 26
Feb
2023

___
Last
login: Fri
Mar
17
12: 28:06
2023
from vdi

-dhcp - 96 - 083.
uio.no
[kielland @ ml6 ~]$ nvidia - smi
Fri
Mar
17
13: 16:11
2023
+-----------------------------------------------------------------------------+
| NVIDIA - SMI
515.86
.01
Driver
Version: 515.86
.01
CUDA
Version: 11.7 |
| -------------------------------+----------------------+----------------------+
| GPU
Name
Persistence - M | Bus - Id
Disp.A | Volatile
Uncorr.ECC |
| Fan
Temp
Perf
Pwr: Usage / Cap | Memory - Usage | GPU - Util
Compute
M. |
| | | MIG
M. |
|= == == == == == == == == == == == == == == == += == == == == == == == == == == = += == == == == == == == == == == = |
| 0
NVIDIA
GeForce...Off | 00000000: 01:00.0
Off | N / A |
| 26 % 46
C
P2
83
W / 250
W | 4483
MiB / 11264
MiB | 34 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 1
NVIDIA
GeForce...Off | 00000000: 23:00.0
Off | N / A |
| 27 % 48
C
P2
95
W / 250
W | 4407
MiB / 11264
MiB | 21 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 2
NVIDIA
GeForce...Off | 00000000: 41:00.0
Off | N / A |
| 26 % 46
C
P2
81
W / 250
W | 4483
MiB / 11264
MiB | 20 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 3
NVIDIA
GeForce...Off | 00000000: 61:00.0
Off | N / A |
| 23 % 41
C
P2
71
W / 250
W | 4483
MiB / 11264
MiB | 39 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 4
NVIDIA
GeForce...Off | 00000000: 81:00.0
Off | N / A |
| 22 % 26
C
P8
1
W / 250
W | 3
MiB / 11264
MiB | 0 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 5
NVIDIA
GeForce...Off | 00000000: A1:00.0
Off | N / A |
| 22 % 29
C
P8
21
W / 250
W | 3
MiB / 11264
MiB | 0 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 6
NVIDIA
GeForce...Off | 00000000: C1:00.0
Off | N / A |
| 22 % 31
C
P8
28
W / 250
W | 5
MiB / 11264
MiB | 0 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+
| 7
NVIDIA
GeForce...Off | 00000000: E1:00.0
Off | N / A |
| 22 % 28
C
P8
25
W / 250
W | 5
MiB / 11264
MiB | 0 % Default |
| | | N / A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes: |
| GPU
GI
CI
PID
Type
Process
name
GPU
Memory |
| ID
ID
Usage |
|= == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == |
| 0
N / A
N / A
1732903
C
python3
3561
MiB |
| 0
N / A
N / A
2349494
C
python
919
MiB |
| 1
N / A
N / A
1733204
C...a / envs / lfdeep / bin / python3
3485
MiB |
| 1
N / A
N / A
2349711
C... / envs / mariaenv / bin / python
919
MiB |
| 2
N / A
N / A
1733216
C...a / envs / lfdeep / bin / python3
3561
MiB |
| 2
N / A
N / A
2349712
C... / envs / mariaenv / bin / python
919
MiB |
| 3
N / A
N / A
1733260
C...a / envs / lfdeep / bin / python3
3561
MiB |
| 3
N / A
N / A
2349713
C... / envs / mariaenv / bin / python
919
MiB |
+-----------------------------------------------------------------------------+
[kielland @ ml6 ~]$ module
load
PyTorch - bundle / 1.10
.0 - MKL - bundle - pre - optimised
[kielland @ ml6 ~]$ ipython
ipython: Command
not found.
[kielland @ ml6 ~]$ python
Python
3.9
.7 | packaged
by
conda - forge | (default, Sep 29 2021, 19:20:46)
[GCC 9.4.0]
on
linux
Type
"help", "copyright", "credits" or "license"
for more information.
    >> > import torch
>> > arr = torch.randn(12).to(device="cuda:6")
>> > arr
tensor([0.2906, -1.9789, -0.3662, 1.4614, -0.1606, 0.5760, 1.0709, -0.3326,
        -0.8738, -0.8319, -0.4780, -1.2366], device='cuda:6')
>> > exit
Use
exit() or Ctrl - D(i.e.EOF)
to
exit
>> > exit()
[kielland @ ml6 ~]$ ls
main07.py
[kielland @ ml6 ~]$ nano
main07.py
[kielland @ ml6 ~]$ python
main07.py
The
number
of
samples in each
data
set
Train
set: 496
samples
Validation
set: 200
samples
Test
set: 300
samples
----------
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth"
to / itf - fi - ml / home / kielland /.cache / torch / hub / checkpoints / resnet18 - f37072fd.pth
100 % |███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | 44.7
M / 44.7
M[00:00 < 00:00, 452
MB / s]
Epoch
1 / 2
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main07.py", line
595, in < module >
losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val = run_training()
File
"/itf-fi-ml/home/kielland/main07.py", line
199, in run_training
train_model(train_loader, val_loader, model, criterion, optimizer, device)
File
"/itf-fi-ml/home/kielland/main07.py", line
136, in train_model
losses = train_epoch(model, train_loader, criterion, device, optimizer)
File
"/itf-fi-ml/home/kielland/main07.py", line
106, in train_epoch
output = model(inputs)  # output tensor = prediction scores (logits) for each class in the output space
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torchvision/models/resnet.py", line
249, in forward
return self._forward_impl(x)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torchvision/models/resnet.py", line
232, in _forward_impl
x = self.conv1(x)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/conv.py", line
446, in forward
return self._conv_forward(input, self.weight, self.bias)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/conv.py", line
442, in _conv_forward
return F.conv2d(input, weight, bias, self.stride,
                RuntimeError: Mismatched
Tensor
types in NNPack
convolutionOutput
[kielland @ ml6
~]$ nano
main07.py
[kielland @ ml6
~]$ python
Python
3.9
.7 | packaged
by
conda - forge | (default, Sep 29 2021, 19:20:46)
[GCC
9.4
.0] on
linux
Type
"help", "copyright", "credits" or "license"
for more information.
         >> > from main07 import run_training
>> > run_training()
Traceback (most recent call last):
    File
"<stdin>", line
1, in < module >
        File
"/itf-fi-ml/home/kielland/main07.py", line
163, in run_training
device = torch.device("cuda" if config['use_cuda'] else "cpu")
NameError: name
'config' is not defined
>> > exit
Use
exit() or Ctrl - D(i.e.EOF)
to
exit
>> > exit()
[kielland @ ml6
~]$ nano
[kielland @ ml6
~]$ nano
main07.py
[kielland @ ml6
~]$ python
Python
3.9
.7 | packaged
by
conda - forge | (default, Sep 29 2021, 19:20:46)
[GCC
9.4
.0] on
linux
Type
"help", "copyright", "credits" or "license"
for more information.
         >> > import torch
>> > from torchvision import models
>> > model = models.ResNet18(pretrained=True)
Traceback (most recent call last):
    File
"<stdin>", line
1, in < module >
        AttributeError: module
'torchvision.models'
has
no
attribute
'ResNet18'
>> > model = models.resnet18(pretrained=True)
             >> > model.to("cuda:6")
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
(layer1): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
(1): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(layer2): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(downsample): Sequential(
    (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(1): BasicBlock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(layer3): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(downsample): Sequential(
    (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(1): BasicBlock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(layer4): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(downsample): Sequential(
    (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(1): BasicBlock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
(fc): Linear(in_features=512, out_features=1000, bias=True)
)
>> > model.device
Traceback(most
recent
call
last):
File
"<stdin>", line
1, in < module >
        File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1177, in __getattr__
raise AttributeError("'{}' object has no attribute '{}'".format(
    AttributeError: 'ResNet'
object
has
no
attribute
'device'
>> > model.conv1.weights.device
Traceback(most
recent
call
last):
File
"<stdin>", line
1, in < module >
        File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1177, in __getattr__
raise AttributeError("'{}' object has no attribute '{}'".format(
    AttributeError: 'Conv2d'
object
has
no
attribute
'weights'
>> > model.conv1
Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
>> > model.conv1.weight.device
device(type='cuda', index=6)
>> > model
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
(layer1): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
(1): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(layer2): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(downsample): Sequential(
    (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(1): BasicBlock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(layer3): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(downsample): Sequential(
    (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(1): BasicBlock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(layer4): Sequential(
    (0): BasicBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(downsample): Sequential(
    (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
(1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(1): BasicBlock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU(inplace=True)
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
)
(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
(fc): Linear(in_features=512, out_features=1000, bias=True)
)
>> > arr = torch.randn(16, 3, 128, 128)
           >> > arr = arr.to("cuda:6")
                      >> > model(arr)
tensor([[-1.0872, 0.3533, -2.0982, ..., 1.1922, 1.9392, 1.7244],
        [-3.3060, -1.0231, -1.6261, ..., -1.7711, 3.1443, 1.2705],
        [-1.2438, 0.5928, -1.6005, ..., -0.4108, -0.3128, 0.6047],
        ...,
        [-1.5928, -0.2145, 0.2872, ..., -2.0284, 1.6945, -1.0563],
        [-1.7464, 2.6987, 4.0992, ..., -2.2205, -2.9126, 2.1012],
        [-2.0322, -4.7796, -2.6680, ..., -5.9735, 1.6493, 2.3005]],
       device='cuda:6', grad_fn= < AddmmBackward0 >)
>> > model(arr).shape
torch.Size([16, 1000])
>> > exit()
[kielland @ ml6
~]$ CUDA_VISIBLE_DEVICES = 4
python
main07.py
CUDA_VISIBLE_DEVICES = 4: Command
not found.
[kielland @ ml6
~]$ export
CUDA_VISIBLE_DEVICES = 4
python
main07.py
export: Command
not found.
[kielland @ ml6
~]$ export
CUDA_VISIBLE_DEVICES = 4
export: Command
not found.
[kielland @ ml6
~]$ export
export: Command
not found.
[kielland @ ml6
~]$ source.
    .. /.bash_logout.bashrc.config /.local /.python_history
                                             . /.bash_profile.cache /.history.nv /.zshrc
[kielland @ ml6
~]$ source.bas
.bash_logout.bash_profile.bashrc
[kielland @ ml6
~]$ source.bashrc
if: Expression
Syntax.
then: Command
not found.
[kielland @ ml6
~]$ source.bashrc
if: Expression
Syntax.
then: Command
not found.
[kielland @ ml6
~]$ nano.bashrc
[kielland @ ml6
~]$ nano
main07.py
[kielland @ ml6
~]$ python
main07.py
The
number
of
samples in each
data
set
Train
set: 496
samples
Validation
set: 200
samples
Test
set: 300
samples
- ---------
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main07.py", line
600, in < module >
          losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val = run_training()
File
"/itf-fi-ml/home/kielland/main07.py", line
192, in run_training
model.to(device)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
899, in to
return self._apply(convert)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
570, in _apply
module._apply(fn)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
593, in _apply
param_applied = fn(param)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
897, in convert
return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/cuda/__init__.py", line
214, in _lazy_init
torch._C._cuda_init()
RuntimeError: No
CUDA
GPUs
are
available
[kielland @ ml6 ~]$ nano
main07.py
[kielland @ ml6 ~]$ python
main07.py
The
number
of
samples in each
data
set
Train
set: 496
samples
Validation
set: 200
samples
Test
set: 300
samples
----------
Epoch
1 / 2
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main07.py", line
600, in < module >
losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val = run_training()
File
"/itf-fi-ml/home/kielland/main07.py", line
204, in run_training
train_model(train_loader, val_loader, model, criterion, optimizer, device)
File
"/itf-fi-ml/home/kielland/main07.py", line
141, in train_model
losses = train_epoch(model, train_loader, criterion, device, optimizer)
File
"/itf-fi-ml/home/kielland/main07.py", line
111, in train_epoch
output = model(inputs)  # output tensor = prediction scores (logits) for each class in the output space
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torchvision/models/resnet.py", line
249, in forward
return self._forward_impl(x)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torchvision/models/resnet.py", line
232, in _forward_impl
x = self.conv1(x)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/conv.py", line
446, in forward
return self._conv_forward(input, self.weight, self.bias)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/conv.py", line
442, in _conv_forward
return F.conv2d(input, weight, bias, self.stride,
                RuntimeError: Mismatched
Tensor
types in NNPack
convolutionOutput
[kielland @ ml6
~]$ nano
main07.py
[kielland @ ml6
~]$ python
main07.py
The
number
of
samples in each
data
set
Train
set: 496
samples
Validation
set: 200
samples
Test
set: 300
samples
- ---------
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main07.py", line
601, in < module >
          losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val = run_training()
File
"/itf-fi-ml/home/kielland/main07.py", line
193, in run_training
print(model.device)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1177, in __getattr__
raise AttributeError("'{}' object has no attribute '{}'".format(
    AttributeError: 'ResNet'
object
has
no
attribute
'device'
[kielland @ ml6
~]$ nano
main07.py
[kielland @ ml6
~]$ python
main07.py
The
number
of
samples in each
data
set
Train
set: 496
samples
Validation
set: 200
samples
Test
set: 300
samples
- ---------
cuda: 0
Epoch
1 / 2
Traceback(most
recent
call
last):
File
"/itf-fi-ml/home/kielland/main07.py", line
601, in < module >
          losses_epochs_train, losses_epochs_val, accuracy_epochs_train, accuracy_epochs_val = run_training()
File
"/itf-fi-ml/home/kielland/main07.py", line
205, in run_training
train_model(train_loader, val_loader, model, criterion, optimizer, device)
File
"/itf-fi-ml/home/kielland/main07.py", line
141, in train_model
losses = train_epoch(model, train_loader, criterion, device, optimizer)
File
"/itf-fi-ml/home/kielland/main07.py", line
111, in train_epoch
output = model(inputs)  # output tensor = prediction scores (logits) for each class in the output space
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torchvision/models/resnet.py", line
249, in forward
return self._forward_impl(x)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torchvision/models/resnet.py", line
232, in _forward_impl
x = self.conv1(x)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/module.py", line
1102, in _call_impl
return forward_call(*input, **kwargs)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/conv.py", line
446, in forward
return self._conv_forward(input, self.weight, self.bias)
File
"/storage/software/PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised/lib/python3.9/site-packages/torch/nn/modules/conv.py", line
442, in _conv_forward
return F.conv2d(input, weight, bias, self.stride,
                RuntimeError: Mismatched
Tensor
types in NNPack
convolutionOutput
[kielland @ ml6
~]$ export
export: Command
not found.
[kielland @ ml6
~]$
[kielland @ ml6 ~]$ nano
main07.py

GNU
nano
2.9
.8
main07.py

import PIL.Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
import itertools

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="8"


# import efficientnet_pytorch as efn

# Ignore the DeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def split_data(reduced_num_samples=False):
    # list of subdirectories containing each class
    subdirs = [x for x in os.listdir(data_dir) if not x.startswith('.')]  # skip hidden files/directories

    file_paths = []  # list to store file paths of images
    labels = []  # labels corresponding to the above list of images

    # reduce the number of samples for testing of code
    num_samples_per_class = 1000 // 6

    # loop over the subdirectories and file paths and labels
    for i, subdir in enumerate(subdirs):
        class_dir = os.path.join(data_dir, subdir)
        image_files = os.listdir(class_dir)
        if reduced_num_samples:
            image_files = image_files[:num_samples_per_class]  # select the first num_samples_per_class images
        image_paths = [os.path.join(class_dir, f) for f in image_files]
        file_paths.extend(image_paths)
        labels.extend([i] * len(image_files))  # label each image with its corresponding class index

    # split data into train, validation, and test sets which are stored as paths in lists
    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=300,
                                                                          random_state=123, stratify=labels)
    [Read 644 lines]

^ G
Get
Help ^ O
Write
Out ^ W
Where
Is ^ K
Cut
Text ^ J
Justify ^ C
Cur
Pos
M - U
Undo
M - A
Mark
Text
M -] To
Bracket ^ B
Back ^ Left
Prev
Word
^ X
Exit ^ R
Read
File ^\ Replace ^ U
Uncut
Text ^ T
To
Linter ^ _
Go
To
Line
M - E
Redo
M - 6
Copy
Text
M - W
WhereIs
Next ^ F
Forward ^ Right
Next
Word
