# NetLib

## Overview
NetLib is a core component of the NN Project, focusing on providing foundational functionalities for neural network operations. It is optimized for performance and scalability, leveraging the power of the CUDA toolkit.

This project is associated with the paper titled "Return of the RNN: Residual Recurrent Networks for Invertible Sentence Embeddings" which provides in-depth explanations of the concepts, methodologies, and findings that underpin this software. The paper can be accessed [here](https://arxiv.org/abs/2303.13570v2).

For detailed documentation, see [NetLib Documentation](https://jjwilkerson.github.io/NetLib/).

## Compatibility

This software has been developed and tested on Linux. While it may work on other UNIX-like systems, its compatibility with non-Linux operating systems (like Windows or macOS) has not been verified. Users are welcome to try running it on other systems, but should be aware that they may encounter issues or unexpected behavior.

## Dependencies
To build NetLib, you'll need the following dependencies:

- **C++ Compiler:** GCC (versions 6.x - 12.2) or an equivalent compiler.
- **CUDA Toolkit:** Version 12.1 - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
- **Additional Libraries (add as include paths only):** 
    - UTF8-CPP (version 2.3.4) [Download UTF8-CPP](https://github.com/nemtrif/utfcpp/tree/v2.3.4).
    - JsonCpp (version 1.8.4) [Download JsonCpp](https://github.com/open-source-parsers/jsoncpp/tree/1.8.4).
    - Boost (version 1.72.0) [Download Boost](https://www.boost.org/users/history/).
        - Build the regex, system, and filesystem modules.
    - CUDA Samples common [Download CUDA Samples](https://github.com/NVIDIA/cuda-samples).
    
## Building NetLib
While NetLib was developed using Eclipse with the Nsight plugin, it's not a strict requirement. You can build it as long as you have the CUDA toolkit installed.

Here are the general steps to build NetLib:

- **Clone the Repository:**
 
```bash
git clone https://github.com/jjwilkerson/NetLib.git
cd NetLib
```

- **Building:**
If the above include paths are configured then NetLib can be easily built (as a library) in Eclipse with the Nsight plugin. Alternatively, you can build it from the command line using Make by executing the following command. It may be necessary to update paths to nvcc and dependencies in Makefile first. You may also need to change the CUDA hardware versions after "arch=" to match your specific GPU.

```bash
make
```

## Usage
The NetLib source folder should be added as an include path in projects that use NetLib, and the built libNetLib.a file should be linked as a library (use name "NetLib").

## Contributing
Contributions to NetLib are welcome. Please ensure to follow the coding standards and submit a pull request for review.

## License
NetLib is licensed under the MIT License. See the LICENSE file for more details.
