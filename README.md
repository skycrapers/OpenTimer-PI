# OpenTimer-PI
Code of "Dynamic Supply Noise Aware Timing Analysis with JIT Machine Learning Integration". 

# System Requirements
+ GNU C++ Compiler v7.3 with -std=c++1z
+ Clang C++ Compiler v6.0 with -std=c++17
+ PyTorch >= 1.6.0

# Basic Usage
```bash
~$ ./bin/ot-shell

ot> cd example/simple
ot> read_celllib osu018_stdcells.lib
ot> read_verilog simple.v   
ot> read_sdc simple.sdc
ot> read_noise_models ../sample/models_trace        # The path of MLP models
ot> report_timing
```

For further usage instructions, please refer to the [OpenTimer](./README_opentimer.md)

# Update MLP Models
1. Place the pre-trained MLP (Multi-Layer Perceptron) weights from PyTorch, saved with the '.pth' extension, in the directory 'OpenTimer-PI/sample.'

2. Execute the script 'trace_pt_script.py' located in the aforementioned directory to generate the Intermediate Representations.

3. Recompile OpenTimer.


# Compile
```bash
~$ cd OpenTimer-PI
~$ mkdir build
~$ cd build
~$ cmake ../
~$ make 
```
After successful build, you can find binaries and libraries in the folders `bin` 
and `lib`, respectively.




# Acknowledgement
We are greatly appreciative of the open-source project [OpenTimer](https://github.com/OpenTimer/OpenTimer/) , upon which we have built and completed this endeavor