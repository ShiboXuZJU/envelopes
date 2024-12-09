#include <iostream>
#include <chrono>
#include "envelopes.cpp"

#define LOG(x) std::cout << x << std::endl

int main()
{
    // double dt = 10;
    // auto start = std::chrono::high_resolution_clock::now();
    // auto g = std::make_shared<Gaussian>(0, 5, 1);
    // auto s = (*g) + (*g >> dt);
    // for (int i = 2; i < 100000; i++)
    // {
    //     (*s) += (*g >> (i * dt));
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}