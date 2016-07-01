#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>
#include <complex>

template<typename T>
class rand_normal {
  static std::mt19937 engine;
  static std::normal_distribution<T> U;
public:
  static void reset() {
    engine=std::mt19937(0);
    U.reset();
  }
  static T get() {
    return U(engine);
  }
};

template<typename T>
std::mt19937 rand_normal<T>::engine(0);

template<typename T>
std::normal_distribution<T> rand_normal<T>::U;


template<typename T>
class rand_normal<std::complex<T>> {
  static std::mt19937 engine;
  static std::normal_distribution<T> U;
public:
  static void reset() {
    engine=std::mt19937(0);
    U.reset();
  }
  static std::complex<T> get() {
    T re(U(engine));
    T im(U(engine));
    return std::complex<T>(re, im);
  }
};

template<typename T>
std::mt19937 rand_normal<std::complex<T>>::engine(0);

template<typename T>
std::normal_distribution<T> rand_normal<std::complex<T>>::U;

#endif
