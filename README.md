# skepsi

A neural network library in c++ aimed at providing a simple, modularized framework for deep learning. 

===== VERSION 0.0.1 =====
- Currently skepsi only works on host machines and only has a very limited memory manager and tensor class.
- More is coming...


### Download and Installation
-----------------------------
First get the repository on your computer with

```sh
git glone https://github.com/Dando18/skepsi
cd skepsi
```

Next copy the make include settings into the head directory and edit them to your preferences.

```sh
cp make.inc-examples/make.inc-standard ./make.inc
vim make.inc # if you want to edit the settings
```

After this simply run `make install` to build and install skepsi. If your prefix (install location) has root priviledge acccess, then you'll need to run with `sudo`.

So the entire script looks like,

```sh
git clone https://github.com/Dando18/skepsi
cd skepsi
cp make.inc-examples/make.inc-standard ./make.inc
sudo make install
```

### Testing 
------------
Skepsi comes with some tester files to make sure everything is working properly. You can build them using the same makefile as for installation. Use the following command to build and run the testers:

```sh
make testing
cd testing
sh ./run_tests.sh
```



_author:_ Daniel Nichols
