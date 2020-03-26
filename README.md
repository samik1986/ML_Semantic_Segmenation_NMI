**SETUP The Environment:**

```
$ sudo apt-get install python-pip python-dev python-virtualenv

$ sudo apt-get install virtualenv

$ virtualenv ~/venv

$ source ~/venv/bin/activate

```
**INSTALL dependencies:**

```
(venv)$ cat requirements.txt | xargs -n 1 pip install

(venv)$ cat requirements3.txt | xargs -n 1 pip3 install
```
