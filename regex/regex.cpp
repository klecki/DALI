#include <pybind11/pybind11.h>
#include <regex>
#include <cstdio>


namespace py = pybind11;


// g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` regex.cpp -o regex`python3-config --extension-suffix`


// pip3.8 install pybind11
// git clone https://github.com/pybind/pybind11
// g++ -O3 -Wall -shared -std=c++11 -fPIC `python3.8 -m pybind11 --includes` -Ipybind11/include/ regex.cpp -o regex_check`python3.8-config --extension-suffix`

// DALI: GLIBCXX: 20180303
// MXNET GLIBCXX: 20200808

int construct_regex() {
    // std::regex header_regex_(R"###(^\{'descr': \'(.*?)\', 'fortran_order': (.*?), 'shape': \((.*?)\), \})###");
    std::printf("GLIBCXX: %d\n",__GLIBCXX__);
    std::regex header_regex_(R"###()###");
    // // std::string txt = "some dummy text, so it is used";
    // std::smatch header_match;
    // bool result = std::regex_search(txt, header_match, header_regex_);
    return header_regex_.mark_count();
}

PYBIND11_MODULE(regex_check, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("construct_regex", &construct_regex, "regex");
}
